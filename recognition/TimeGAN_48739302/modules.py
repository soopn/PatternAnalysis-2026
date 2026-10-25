import copy

import torch
import torch.optim as optim
import torch.nn as nn


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
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()

        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        d, _ = self.rnn(h)
        d = self.fc(d)
        return d


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
    ):
        super().__init__()

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

        self.embedder = Embedder(input_dim, hidden_dim, num_layers)
        self.recoverer = Recoverer(hidden_dim, input_dim, num_layers)
        self.generator = Generator(noise_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, num_layers=1)
        self.discriminator = Discriminator(hidden_dim, num_layers)

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

        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

        self.optimizer_er = optim.Adam(
            list(self.embedder.parameters())
            + list(self.recoverer.parameters()),
            lr=learning_rate,
        )

        self.optimizer_s = optim.Adam(
            self.supervisor.parameters(),
            lr=learning_rate,
        )

        self.optimizer_g = optim.Adam(
            list(self.generator.parameters())
            + list(self.supervisor.parameters()),
            lr=learning_rate,
        )

        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
        )

    def sample_noise(self, batch_size, seq_len, device):
        """Uniform[0, 1] noise with two components:
        - per-step channels: AR(1) in Gaussian space mapped to uniform
          marginals, so neighbouring steps get similar noise (rho = noise_rho)
          instead of independent draws, which made the output flicker;
        - per-sequence channels: one draw held constant over the whole
          sequence, so each sample can have its own regime.
        """
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
        den = torch.sqrt((a * a).mean(dim=(0, 1)) * (b * b).mean(dim=(0, 1)))
        return num / (den + eps)

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
        loss = self.mse(x_tilde, x)

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

    # generator + supervisor update
    def train_generator_step(self, x):
        batch_size, seq_len, _ = x.shape

        with torch.no_grad():
            h = self.embedder(x)

        z = self.sample_noise(batch_size, seq_len, x.device)

        # Freeze discriminator weights, but retain gradients
        # through it into the generated latent sequence
        self.set_requires_grad(self.discriminator, False)
        self.optimizer_g.zero_grad(set_to_none=True)

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

        # Match feature means and standard deviations at every timestep
        # (statistics over the batch axis only -> [T, D], as in the
        # reference TimeGAN).
        x_hat = self.recoverer(h_hat)

        mean_real = x.mean(dim=0)
        mean_fake = x_hat.mean(dim=0)

        std_real = torch.sqrt(x.var(dim=0, unbiased=False) + 1e-6)
        std_fake = torch.sqrt(x_hat.var(dim=0, unbiased=False) + 1e-6)

        loss_moments = (
            torch.abs(mean_real - mean_fake).mean()
            + torch.abs(std_real - std_fake).mean()
        )

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

        # standard weighting
        loss_g = (
            self.adv_weight * (loss_adv + self.gamma * loss_adv_e)
            + 100.0 * torch.sqrt(loss_supervised + 1e-8)
            + 100.0 * loss_moments
            + self.diff_weight * loss_diff
            + self.corr_weight * loss_corr
            + self.ac_weight * loss_ac
            + self.lat_ac_weight * loss_ac_h
        )

        loss_g.backward()
        self.optimizer_g.step()

        self.set_requires_grad(self.discriminator, True)

        return loss_g.detach().item()

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
        if loss_d.item() > self.d_threshold:
            loss_d.backward()
            self.optimizer_d.step()

        return loss_d.detach().item()

    def train_joint_step(self, x):
        """Adversarial training step"""

        # update generator and embedder/recoverer twice per discriminator step
        for _ in range(self.gen_steps):
            g_loss = self.train_generator_step(x)
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
