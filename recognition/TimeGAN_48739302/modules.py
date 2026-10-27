import copy

import torch
import torch.optim as optim
import torch.nn as nn


def initialise_recurrent(module):
    """Reference-style initialisation, applied only to fresh networks."""
    if isinstance(module, nn.GRU):
        for name, parameter in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class Embedder(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layers=2, activation_fun=nn.Sigmoid()
    ):
        super().__init__()

        self.rnn_cell = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act_fun = activation_fun

    def forward(self, x):
        h, _ = self.rnn_cell(x)
        h = self.fc(h)
        return self.act_fun(h)


class Recoverer(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super().__init__()

        self.rnn_cell = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.rnn_cell(x)
        x = self.fc(x)
        return x


class Generator(nn.Module):
    def __init__(
        self, noise_dim, hidden_dim, num_layers=2, activation_fun=nn.Sigmoid()
    ):
        super().__init__()

        self.rnn_cell = nn.GRU(
            noise_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act_fun = activation_fun

    def forward(self, x):
        h, _ = self.rnn_cell(x)
        h = self.fc(h)
        h = self.act_fun(h)
        return h


class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers=1, activation_fun=nn.Sigmoid()):
        super().__init__()

        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act_fun = activation_fun

    def forward(self, h):
        s, _ = self.rnn(h)
        s = self.fc(s)
        s = self.act_fun(s)
        return s


class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, input_mode="raw"):
        super().__init__()
        if input_mode not in ("raw", "standardised"):
            raise ValueError("Unknown discriminator input mode")
        self.input_mode = input_mode
        if input_mode == "standardised":
            self.register_buffer("input_center", torch.zeros(hidden_dim))
            self.register_buffer("input_scale", torch.ones(hidden_dim))
            self.register_buffer("input_ready", torch.zeros(1))

        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        if self.input_mode == "standardised" and self.input_ready.item():
            h = (h - self.input_center) / self.input_scale
        d, _ = self.rnn(h)
        d = self.fc(d)
        return d

    @torch.no_grad()
    def fit_input_statistics(self, real_latents):
        """Fit once on real training latents, preserving the current function.

        Freeze these statistics thereafter; fake samples and validation never
        update them. The first GRU input affine map absorbs the coordinate
        change so switching on conditioning does not jump the predictions.
        """
        if self.input_mode == "raw" or self.input_ready.item():
            return
        center = real_latents.mean((0, 1))
        scale = real_latents.std((0, 1), unbiased=False).clamp_min(1e-3)
        original = self.rnn.weight_ih_l0.clone()
        self.rnn.weight_ih_l0.mul_(scale[None, :])
        self.rnn.bias_ih_l0.add_(original @ center)
        self.input_center.copy_(center)
        self.input_scale.copy_(scale)
        self.input_ready.fill_(1)


class TimeGAN(nn.Module):
    """TimeGAN model wrapper"""

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        noise_dim=64,
        num_layers=2,
        seq_len=100,
        learning_rate=1e-3,
        gamma=1.0,
        noise_rho=0.7,
        diff_weight=100.0,
        corr_weight=50.0,
        ac_weight=50.0,
        adv_weight=10.0,
        d_threshold=0.6,
        gen_steps=2,
        lat_ac_weight=50.0,
        ema_decay=0.999,
        within_weight=50.0,
        reconstruction_event_weight=1.0,
        noise_mode="correlated",
        adam_beta1=0.9,
        moment_mode="per_timestep",
        initialisation="default",
        generator_auxiliary_scale=1.0,
        discriminator_input="raw",
        supervisor_loss_weight=None,
    ):
        super().__init__()
        if noise_mode not in ("correlated", "independent"):
            raise ValueError("Unknown noise_mode")
        if moment_mode not in ("per_timestep", "pooled"):
            raise ValueError("Unknown moment_mode")
        if initialisation not in ("default", "xavier_orthogonal"):
            raise ValueError("Unknown initialisation")
        if generator_auxiliary_scale < 0:
            raise ValueError("generator_auxiliary_scale must be nonnegative")
        if supervisor_loss_weight is not None and supervisor_loss_weight < 0:
            raise ValueError("supervisor_loss_weight must be nonnegative or None")
        # Save constructor settings so inference uses the training noise law.
        self.model_config = {
            "input_dim": input_dim, "hidden_dim": hidden_dim,
            "noise_dim": noise_dim, "num_layers": num_layers,
            "seq_len": seq_len, "learning_rate": learning_rate,
            "gamma": gamma, "noise_rho": noise_rho,
            "diff_weight": diff_weight, "corr_weight": corr_weight,
            "ac_weight": ac_weight, "adv_weight": adv_weight,
            "d_threshold": d_threshold, "gen_steps": gen_steps,
            "lat_ac_weight": lat_ac_weight, "ema_decay": ema_decay,
            "within_weight": within_weight,
            "reconstruction_event_weight": reconstruction_event_weight,
            "noise_mode": noise_mode, "adam_beta1": adam_beta1,
            "moment_mode": moment_mode, "initialisation": initialisation,
            "generator_auxiliary_scale": generator_auxiliary_scale,
            "discriminator_input": discriminator_input,
            "supervisor_loss_weight": supervisor_loss_weight,
        }
        self.noise_mode = noise_mode
        self.moment_mode = moment_mode
        self.generator_auxiliary_scale = generator_auxiliary_scale
        # Effective coefficient, independent of the global auxiliary scale.
        # None preserves the old 100 * auxiliary_scale behaviour for checkpoints.
        self.supervisor_loss_weight = supervisor_loss_weight

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        self.gamma = gamma
        self.noise_rho = noise_rho  # autocorrelation of the per-step noise
        self.diff_weight = diff_weight  # weight of the roughness-matching loss
        self.corr_weight = corr_weight  # weight of the correlation-matching loss
        self.ac_weight = ac_weight  # weight of the autocorrelation-matching loss
        self.adv_weight = adv_weight  # weight of the adversarial terms
        self.d_threshold = d_threshold  # D only updates while its loss exceeds this
        self.gen_steps = gen_steps
        self.lat_ac_weight = lat_ac_weight
        self.within_weight = within_weight  # within-window std / correlation matching
        self.reconstruction_event_weight = reconstruction_event_weight

        self.embedder = Embedder(input_dim, hidden_dim, num_layers)
        self.recoverer = Recoverer(hidden_dim, input_dim, num_layers)
        self.generator = Generator(noise_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, num_layers=1)
        self.discriminator = Discriminator(hidden_dim, num_layers, input_mode=discriminator_input)
        if initialisation == "xavier_orthogonal":
            self.apply(initialise_recurrent)

        # ema for generation path of generator, supervisor, recoverer 
        self.ema_decay = ema_decay
        self.ema = nn.ModuleDict(
            {
                "generator": copy.deepcopy(self.generator),
                "supervisor": copy.deepcopy(self.supervisor),
                "recoverer": copy.deepcopy(self.recoverer),
            }
        )
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.register_buffer("ema_ready", torch.zeros(1))

        self.register_buffer("feature_mean", torch.zeros(input_dim), persistent=False)
        self.register_buffer("feature_std", torch.ones(input_dim), persistent=False)
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.optimizer_er = optim.Adam(
            list(self.embedder.parameters())
            + list(self.recoverer.parameters()),
            lr=learning_rate,
            betas=(adam_beta1, 0.999),
        )

        self.optimizer_s = optim.Adam(
            self.supervisor.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, 0.999),
        )

        self.optimizer_g = optim.Adam(
            list(self.generator.parameters())
            + list(self.supervisor.parameters()),
            lr=learning_rate,
            betas=(adam_beta1, 0.999),
        )

        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            betas=(adam_beta1, 0.999),
        )

    def sample_noise(self, batch_size, seq_len, device):
        """Uniform[0, 1] noise, independent or with two correlated components:
        - per-step channels: AR(1) in Gaussian space mapped to uniform
          marginals, so neighbouring steps get similar noise (rho = noise_rho)
          instead of independent draws, which made the output flicker;
        - per-sequence channels: one draw held constant over the whole
          sequence, so each sample can have its own regime.
        """
        if self.noise_mode == "independent":
            return torch.rand(batch_size, seq_len, self.noise_dim, device=device)
        half = self.noise_dim // 2
        rho = self.noise_rho
        scale = (1.0 - rho * rho) ** 0.5

        eps = torch.randn(batch_size, seq_len, half, device=device)
        g = torch.empty_like(eps)
        g[:, 0] = eps[:, 0]
        for t in range(1, seq_len):
            g[:, t] = rho * g[:, t - 1] + scale * eps[:, t]
        z_step = 0.5 * (1.0 + torch.erf(g / 2.0 ** 0.5))

        z_seq = torch.rand(
            batch_size, 1, self.noise_dim - half, device=device
        ).expand(-1, seq_len, -1)

        return torch.cat([z_step, z_seq], dim=-1)

    @staticmethod
    def corr_matrix(x, eps=1e-3):
        """Feature correlation matrix [D, D] pooled over batch and time."""
        x = x.reshape(-1, x.shape[-1])
        x = x - x.mean(dim=0, keepdim=True)
        x = x / (x.std(dim=0, unbiased=False, keepdim=True) + eps)
        return x.T @ x / x.shape[0]

    @staticmethod
    def autocorr(x, lag, eps=1e-3):
        """Lag-`lag` autocorrelation per feature, pooled over batch and time.
        x: [B, T, D] -> [D]. Alternating series give ~-1, persistent ones ~+1."""
        a, b = x[:, :-lag], x[:, lag:]
        a = a - a.mean(dim=(0, 1), keepdim=True)
        b = b - b.mean(dim=(0, 1), keepdim=True)
        num = (a * b).mean(dim=(0, 1))
        den = torch.sqrt(((a * a).mean(dim=(0, 1)) + eps)
                         * ((b * b).mean(dim=(0, 1)) + eps))
        return num / den

    def moment_loss(self, real, fake):
        dims = (0, 1) if self.moment_mode == "pooled" else (0,)
        std_real = torch.sqrt(real.var(dim=dims, unbiased=False) + 1e-6)
        std_fake = torch.sqrt(fake.var(dim=dims, unbiased=False) + 1e-6)
        return ((real.mean(dim=dims) - fake.mean(dim=dims)).abs().mean()
                + (std_real - std_fake).abs().mean())

    @staticmethod
    def event_losses(real, fake, feature_std):
        """Distribution losses in original feature units; no hard rounding.

        A soft change indicator separates event frequency from jump size.
        Sorted samples compare empirical distributions without paired windows.
        """
        vol = [i for i in range(40) if i % 4 in (1, 3)]
        r, f = real * feature_std, fake * feature_std
        dr, df = r[:, 1:, vol] - r[:, :-1, vol], f[:, 1:, vol] - f[:, :-1, vol]
        # Real log-volume changes are exactly zero at held levels.
        gr, gf = 1 - torch.exp(-dr.abs() / 0.05), 1 - torch.exp(-df.abs() / 0.05)
        frequency = (gr.mean((0, 1)) - gf.mean((0, 1))).abs().mean()
        jumps = (dr.abs().reshape(-1, 20).sort(0).values
                 - df.abs().reshape(-1, 20).sort(0).values).abs().mean()
        coordination = (gr.sum(-1).flatten().sort().values
                        - gf.sum(-1).flatten().sort().values).abs().mean() / 20
        within = (r.std(1, unbiased=False).sort(0).values
                  - f.std(1, unbiased=False).sort(0).values).abs().mean()
        timing = (r[..., 41].flatten().sort().values
                  - f[..., 41].flatten().sort().values).abs().mean()
        return frequency + jumps + coordination + within + timing

    @staticmethod
    def reconstruction_event_losses(real, reconstructed, feature_std):
        """Paired volume-event losses in original log-volume units.

        Unlike generator distribution matching, reconstruction must preserve
        the location and sign of each event. Balance held and changed entries
        so the many unchanged levels cannot hide attenuated jumps.
        """
        vol = [i for i in range(40) if i % 4 in (1, 3)]
        dr = torch.diff(real[..., vol], dim=1) * feature_std[vol]
        df = torch.diff(reconstructed[..., vol], dim=1) * feature_std[vol]
        if dr.numel() == 0:
            zero = reconstructed.sum() * 0.0
            return dict.fromkeys(("frequency", "jumps", "coordination", "alignment"), zero)
        gr = -torch.expm1(-dr.abs() / 0.05)
        gf = -torch.expm1(-df.abs() / 0.05)
        changed = (dr.abs() > 1e-6).to(dr.dtype)
        error = (dr - df).abs()
        jump_error = (error * changed).sum() / changed.sum().clamp_min(1)
        hold_error = (error * (1 - changed)).sum() / (1 - changed).sum().clamp_min(1)
        return {
            "frequency": (gr.mean((0, 1)) - gf.mean((0, 1))).abs().mean(),
            "jumps": jump_error + hold_error,
            "coordination": (gr.sum(-1) - gf.sum(-1)).abs().mean() / len(vol),
            "alignment": (gr - gf).abs().mean(),
        }

    @staticmethod
    def volatility_loss(real, fake, feature_mean, feature_std):
        # Invert the asinh midprice feature before measuring volatility.
        ar = torch.sinh(real[..., 40:41] * feature_std[40] + feature_mean[40]).abs()
        af = torch.sinh((fake[..., 40:41] * feature_std[40] + feature_mean[40]).clamp(-10, 10)).abs()
        return torch.stack([
            (TimeGAN.autocorr(ar[:, 1:], k, eps=1e-6)
             - TimeGAN.autocorr(af[:, 1:], k, eps=1e-6)).abs().mean()
            for k in range(1, min(11, real.shape[1] - 1))
        ]).mean()

    @staticmethod
    def set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def reconstruct(self, x):
        h = self.embedder(x)
        return self.recoverer(h)

    @torch.no_grad()
    def init_ema(self):
        """Copy the current weights into the EMA copies"""
        for name in self.ema:
            self.ema[name].load_state_dict(getattr(self, name).state_dict())
        self.ema_ready.fill_(1.0)

    @torch.no_grad()
    def update_ema(self):
        d = self.ema_decay
        for name, shadow in self.ema.items():
            for p_ema, p in zip(
                shadow.parameters(), getattr(self, name).parameters()
            ):
                p_ema.mul_(d).add_(p.detach(), alpha=1.0 - d)

    def nets(self, use_ema=True):
        """(generator, supervisor, recoverer) used for sampling."""
        if use_ema and self.ema_ready.item() > 0:
            return (
                self.ema["generator"],
                self.ema["supervisor"],
                self.ema["recoverer"],
            )
        return self.generator, self.supervisor, self.recoverer

    def generate(self, batch_size, seq_len=None, use_ema=True):
        if seq_len is None:
            seq_len = self.seq_len

        device = next(self.parameters()).device
        z = self.sample_noise(batch_size, seq_len, device)

        generator, supervisor, recoverer = self.nets(use_ema)
        e_hat = generator(z)
        h_hat = supervisor(e_hat)
        return recoverer(h_hat)

    def pretrain_autoencoder(self, x):
        self.optimizer_er.zero_grad(set_to_none=True)

        x_tilde = self.reconstruct(x)
        events = self.reconstruction_event_losses(x, x_tilde, self.feature_std)
        loss = self.mse(x_tilde, x) + self.reconstruction_event_weight * sum(events.values())

        loss.backward()
        self.optimizer_er.step()

        return loss.detach().item()

    def pretrain_supervisor(self, x):
        with torch.no_grad():
            h = self.embedder(x)

        self.optimizer_s.zero_grad(set_to_none=True)

        h_hat = self.supervisor(h)

        loss = self.mse(
            h_hat[:, :-1, :],
            h[:, 1:, :],
        )

        loss.backward()
        self.optimizer_s.step()

        return loss.detach().item()

    def generator_objectives(self, x, z=None):
        """Weighted terms used by G/S, shared by training and gradient audits."""
        batch_size, seq_len, _ = x.shape

        with torch.no_grad():
            h = self.embedder(x)

        self.discriminator.fit_input_statistics(h)
        if z is None:
            z = self.sample_noise(batch_size, seq_len, x.device)

        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)

        y_fake = self.discriminator(h_hat)
        y_fake_e = self.discriminator(e_hat)

        # Fool discriminator at both latent stages.
        loss_adv = self.bce(y_fake, torch.ones_like(y_fake))
        loss_adv_e = self.bce(y_fake_e, torch.ones_like(y_fake_e))

        # Match one-step latent dynamics.
        h_supervised = self.supervisor(h)
        loss_supervised = self.mse(
            h_supervised[:, :-1, :],
            h[:, 1:, :],
        )

        # Match moments pooled over batch/time or separately per timestep.
        x_hat = self.recoverer(h_hat)

        loss_moments = self.moment_loss(x, x_hat)

        # Match step-to-step roughness (mean |difference| per feature), in
        # feature space and in latent space so the supervisor is constrained
        # as well. Moments alone cannot see temporal flicker.
        dx_real = (x[:, 1:] - x[:, :-1]).abs().mean(dim=(0, 1))
        dx_fake = (x_hat[:, 1:] - x_hat[:, :-1]).abs().mean(dim=(0, 1))
        dh_real = (h[:, 1:] - h[:, :-1]).abs().mean(dim=(0, 1))
        dh_fake = (h_hat[:, 1:] - h_hat[:, :-1]).abs().mean(dim=(0, 1))
        loss_diff = (dx_real - dx_fake).abs().mean() + (
            dh_real - dh_fake
        ).abs().mean()

        # Match cross-feature correlations. Means and stds cannot constrain
        # e.g. that ask and bid gaps move together, which sets the spread's
        # variance.
        loss_corr = (
            self.corr_matrix(x) - self.corr_matrix(x_hat)
        ).abs().mean()

        # Match autocorrelation (lags 1, 2). Mean |difference| alone can be
        # satisfied by a perfectly alternating series (autocorr -1), which is
        # what the generator did for the inter-event time.
        loss_ac = sum(
            (self.autocorr(x, k) - self.autocorr(x_hat, k)).abs().mean()
            for k in (1, 2)
        )

        # # match the latent autocorrelation directly
        # latents have a tiny variance so small eps
        loss_ac_h = sum(
            (self.autocorr(h, k, eps=1e-6) - self.autocorr(h_hat, k, eps=1e-6))
            .abs()
            .mean()
            for k in (1, 2)
        )

        # Within window statistics. Per-timestep moments only see variance
        # across windows, not how much a series moves inside one window
        xc = x - x.mean(dim=1, keepdim=True)
        xc_hat = x_hat - x_hat.mean(dim=1, keepdim=True)
        std_w = torch.sqrt(xc.pow(2).mean(dim=1) + 1e-6).mean(dim=0)
        std_w_hat = torch.sqrt(xc_hat.pow(2).mean(dim=1) + 1e-6).mean(dim=0)
        loss_within = (std_w - std_w_hat).abs().mean() + (
            self.corr_matrix(xc) - self.corr_matrix(xc_hat)
        ).abs().mean()

        # standard weighting
        loss_events = self.event_losses(x, x_hat, self.feature_std)
        loss_volatility = self.volatility_loss(x, x_hat, self.feature_mean, self.feature_std)
        terms = {
            "adversarial": self.adv_weight * (loss_adv + self.gamma * loss_adv_e),
            "supervised": 100.0 * torch.sqrt(loss_supervised + 1e-8),
            "moments": 100.0 * loss_moments,
            "difference": self.diff_weight * loss_diff,
            "correlation": self.corr_weight * loss_corr,
            "autocorrelation": self.ac_weight * loss_ac,
            "latent_autocorrelation": self.lat_ac_weight * loss_ac_h,
            "within": self.within_weight * loss_within,
            "events": 25.0 * loss_events,
            "volatility": 25.0 * loss_volatility,
        }
        weighted = {name: value if name == "adversarial" else self.generator_auxiliary_scale * value
                    for name, value in terms.items()}
        if self.supervisor_loss_weight is not None:
            weighted["supervised"] = self.supervisor_loss_weight * torch.sqrt(loss_supervised + 1e-8)
        return weighted

    @property
    def effective_supervisor_loss_weight(self):
        return (100.0 * self.generator_auxiliary_scale if self.supervisor_loss_weight is None
                else self.supervisor_loss_weight)

    def generator_gradient_diagnostics(self, terms, detailed=False):
        """Measure actual weighted gradients without changing parameters/.grad."""
        generator = list(self.generator.parameters())
        supervisor = list(self.supervisor.parameters())
        parameters = generator + supervisor
        split = sum(p.numel() for p in generator)

        def vector(loss):
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            result = torch.cat([
                (g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                for p, g in zip(parameters, gradients)
            ])
            if not torch.isfinite(result).all():
                raise RuntimeError("Nonfinite generator gradient in diagnostic")
            return result

        auxiliary = sum(value for name, value in terms.items() if name != "adversarial")
        vectors = {"adversarial": vector(terms["adversarial"]), "auxiliary": vector(auxiliary)}
        if detailed:
            vectors.update({name: vector(value) for name, value in terms.items()
                            if name != "adversarial"})
        groups = {"generator": slice(0, split), "supervisor": slice(split, None),
                  "combined": slice(None)}
        report = {"weighted_losses": {name: value.detach().item() for name, value in terms.items()},
                  "gradients": {}}
        for name, indices in groups.items():
            adv, aux = vectors["adversarial"][indices], vectors["auxiliary"][indices]
            an, xn = adv.norm().item(), aux.norm().item()
            report["gradients"][name] = {
                "adversarial_norm": an, "auxiliary_norm": xn,
                "adversarial_to_auxiliary_ratio": an / xn if xn else None,
                "cosine_adv_aux": max(-1.0, min(1.0, torch.dot(adv, aux).item() / (an * xn))) if an and xn else None,
                "total_norm": (adv + aux).norm().item(),
                "component_norms": {key: value[indices].norm().item() for key, value in vectors.items()},
            }
            if detailed:
                report["gradients"][name]["component_cosine_to_adversarial"] = {
                    key: max(-1.0, min(1.0, torch.dot(adv, value[indices]).item() / (an * value[indices].norm().item())))
                    if an and value[indices].norm().item() else None
                    for key, value in vectors.items()
                }
        return report

    # generator + supervisor update
    def train_generator_step(self, x, profile_gradients=False):
        flags = [p.requires_grad for p in self.discriminator.parameters()]
        self.set_requires_grad(self.discriminator, False)
        try:
            self.optimizer_g.zero_grad(set_to_none=True)
            terms = self.generator_objectives(x)
            if profile_gradients:
                self.last_generator_diagnostics = self.generator_gradient_diagnostics(terms)
            loss_g = sum(terms.values())
            loss_g.backward()
            self.optimizer_g.step()
            return loss_g.detach().item()
        finally:
            for parameter, flag in zip(self.discriminator.parameters(), flags):
                parameter.requires_grad_(flag)

    # embedder + recovery update
    def train_embedder_step(self, x):
        self.optimizer_er.zero_grad(set_to_none=True)

        # allow gradients through the supervisor into h,
        # without updating supervisor parameters
        self.set_requires_grad(self.supervisor, False)

        h = self.embedder(x)
        x_tilde = self.recoverer(h)

        h_supervised = self.supervisor(h)

        loss_reconstruction = self.mse(x_tilde, x)
        loss_supervised = self.mse(
            h_supervised[:, :-1, :],
            h[:, 1:, :],
        )

        loss_er = (
            10.0 * torch.sqrt(loss_reconstruction + 1e-8)
            + 0.1 * loss_supervised
            + self.reconstruction_event_weight * sum(
                self.reconstruction_event_losses(x, x_tilde, self.feature_std).values()
            )
        )

        loss_er.backward()
        self.optimizer_er.step()

        self.set_requires_grad(self.supervisor, True)

        return loss_er.detach().item()

    def train_discriminator_step(self, x):
        batch_size, seq_len, _ = x.shape

        with torch.no_grad():
            h = self.embedder(x)

            z = self.sample_noise(batch_size, seq_len, x.device)

            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)

        self.discriminator.fit_input_statistics(h)
        self.optimizer_d.zero_grad(set_to_none=True)

        y_real = self.discriminator(h.detach())
        y_fake = self.discriminator(h_hat.detach())
        y_fake_e = self.discriminator(e_hat.detach())

        loss_real = self.bce(y_real, torch.ones_like(y_real))
        loss_fake = self.bce(y_fake, torch.zeros_like(y_fake))
        loss_fake_e = self.bce(y_fake_e, torch.zeros_like(y_fake_e))

        loss_d = loss_real + loss_fake + self.gamma * loss_fake_e

        # mean P(real) the discriminator assigns, for logging
        self.last_d_probs = (
            torch.sigmoid(y_real).mean().item(),
            torch.sigmoid(y_fake).mean().item(),
        )

        # Bound discriminator training: skip the update while D is already
        # winning clearly (sum of three BCE terms; chance level is ~2.08).
        self.last_d_updated = loss_d.item() > self.d_threshold
        if self.last_d_updated:
            loss_d.backward()
            self.optimizer_d.step()

        return loss_d.detach().item()

    def train_joint_step(self, x, profile_gradients=False):
        """Adversarial training step"""

        # update generator and embedder/recoverer twice per discriminator step
        for step in range(self.gen_steps):
            g_loss = self.train_generator_step(x, profile_gradients=profile_gradients and step == self.gen_steps - 1)
            er_loss = self.train_embedder_step(x)
        d_loss = self.train_discriminator_step(x)
        if self.ema_ready.item() > 0:
            self.update_ema()

        return {
            "generator": g_loss,
            "embedder_recovery": er_loss,
            "discriminator": d_loss,
            "d_real": self.last_d_probs[0],
            "d_fake": self.last_d_probs[1],
        }
