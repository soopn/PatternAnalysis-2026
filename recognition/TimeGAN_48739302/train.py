"""
Core investigations:
    - does TimeGAN learn and enforce the structural constraints compared
      to a standard baseline generator
    - does it reproduce authentic volatility clustering or noise/mode collapse

Recommendations:
    - compute KL divergence on spreads/returns (target KL <= 0.1)
    - evaluate depth heatmap structural similarity (price levels x time steps)
      (target SSIM > 0.6)
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset

from dataset import N_FEATURES, LOBFeatureTransform, LOBSTERDataset
from modules import TimeGAN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 100
BATCH_SIZE = 128
RAND_SEED = 69

# 09:30 -> 13:00
TRAIN_START, TRAIN_END = 34_200, 46_800
# 13:00 -> 14:30
EVAL_START, EVAL_END = 46_800, 52_200

AUTOENCODER_EPOCHS = 100
SUPERVISOR_EPOCHS = 100
JOINT_EPOCHS = 200

# Windows use stride 1, so they overlap almost completely. An "epoch" here is
# a random subset of windows instead of every one of them.
STEPS_PER_EPOCH = 200
EVAL_WINDOWS = 20 * BATCH_SIZE  # strided subset of the eval set
FIT_STRIDE = 10  # subsample for fitting the scaler
CHECKPOINT_EVERY = 10
MIN_BEST_EPOCH = (
    30  # ignore very early epochs when picking the 'best' checkpoint
)

# Edit these settings before pasting this file into a Jupyter cell.
CKPT_DIR = Path("checkpoints/reconstruction_events_v1")
RECONSTRUCTION_EVENT_WEIGHT = 1.0  # 0.0 gives the MSE-only reconstruction baseline

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RAND_SEED)


def strided_subset(dataset, max_items):
    """Evenly spaced windows across the whole period."""
    stride = max(1, (len(dataset) + max_items - 1) // max_items)
    return Subset(dataset, range(0, len(dataset), stride))


def make_loaders(train_set, eval_set):
    sampler = RandomSampler(
        train_set,
        replacement=False,
        num_samples=min(len(train_set), STEPS_PER_EPOCH * BATCH_SIZE),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        drop_last=True,
    )
    eval_loader = DataLoader(
        strided_subset(eval_set, EVAL_WINDOWS),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    return train_loader, eval_loader


def get_batches(loader, transform):
    """Yield standardised features on DEVICE."""
    for batch in loader:
        raw = batch["orderbook"].to(DEVICE, non_blocking=True)
        msg = batch["message"].to(DEVICE, non_blocking=True)
        yield transform.normalise(transform.transform_raw(raw, msg))


def raw_feature_batches(loader, transform):
    for batch in loader:
        yield transform.transform_raw(batch["orderbook"], batch["message"])


def moment_gap(real, fake):
    """Per-timestep mean/std gap (as in the generator's moment loss) plus a
    roughness, correlation-matrix and autocorrelation gaps, all in standardised units.
    The extra terms stop a flickering or decorrelated generator from being
    picked as the 'best' checkpoint."""
    d_mean = (real.mean(0) - fake.mean(0)).abs().mean()
    d_std = (real.std(0) - fake.std(0)).abs().mean()
    dx_real = (real[:, 1:] - real[:, :-1]).abs().mean(dim=(0, 1))
    dx_fake = (fake[:, 1:] - fake[:, :-1]).abs().mean(dim=(0, 1))
    d_rough = (dx_real - dx_fake).abs().mean()
    d_corr = (
        (TimeGAN.corr_matrix(real) - TimeGAN.corr_matrix(fake)).abs().mean()
    )
    d_ac = sum(
        (TimeGAN.autocorr(real, k) - TimeGAN.autocorr(fake, k)).abs().mean()
        for k in (1, 2)
    )
    rc = real - real.mean(dim=1, keepdim=True)
    fc = fake - fake.mean(dim=1, keepdim=True)
    d_within = (
        (
            rc.pow(2).mean(dim=1).sqrt().mean(dim=0)
            - fc.pow(2).mean(dim=1).sqrt().mean(dim=0)
        )
        .abs()
        .mean()
    )
    return (d_mean + d_std + d_rough + d_corr + d_ac + d_within).item()


@torch.no_grad()
def run_validation(model, loader, transform, with_generation=False):
    """Reconstruction MSE on real eval data and, optionally, how close the
    generated feature moments are to the real ones."""
    model.eval()

    total, n = 0.0, 0
    event_totals = {}
    real = []
    for x in get_batches(loader, transform):
        x_hat = model.reconstruct(x)
        total += F.mse_loss(x_hat, x).item() * x.size(0)
        for name, value in model.reconstruction_event_losses(x, x_hat, model.feature_std).items():
            event_totals[name] = event_totals.get(name, 0.0) + value.item() * x.size(0)
        n += x.size(0)
        if with_generation:
            real.append(x)

    gap = None
    if with_generation and real:
        real = torch.cat(real)
        devices = [DEVICE.index or 0] if DEVICE.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(RAND_SEED)
            fake = torch.cat(
                [
                    model.generate(min(256, len(real) - i), real.size(1) + 20)[
                        :, 20:
                    ]
                    for i in range(0, len(real), 256)
                ]
            )

        gap = moment_gap(real, fake)
        gap += model.event_losses(real, fake, model.feature_std).item()
        gap += model.volatility_loss(
            real, fake, model.feature_mean, model.feature_std
        ).item()

    model.last_validation_events = {name: value / max(n, 1) for name, value in event_totals.items()}
    model.train()
    return total / max(n, 1), gap


def save_checkpoint(model, transform, path, **extra):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": transform.mean,
            "feature_std": transform.std,
            "validation_version": "events_v1_burn20",
            "loss_weights": {"events": 25.0, "volatility": 25.0,
                             "reconstruction_events": model.reconstruction_event_weight},
            "reconstruction_version": "paired_volume_events_v1",
            **extra,
        },
        path,
    )


def main(model, train_loader, eval_loader, transform):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # embedder + recoverer autoencoder
    for epoch in range(AUTOENCODER_EPOCHS):
        model.train()
        losses = [
            model.pretrain_autoencoder(x)
            for x in get_batches(train_loader, transform)
        ]
        val, _ = run_validation(model, eval_loader, transform)
        print(
            f"[Autoencoder] {epoch + 1:03d} "
            f"train={np.mean(losses):.6f} val_reconstruction={val:.6f}"
            + " " + " ".join(f"val_event_{k}={v:.5f}" for k, v in model.last_validation_events.items())
        )

    # supervisor learns latent temporal dynamics
    for epoch in range(SUPERVISOR_EPOCHS):
        model.train()
        losses = [
            model.pretrain_supervisor(x)
            for x in get_batches(train_loader, transform)
        ]
        print(f"[Supervisor] {epoch + 1:03d} train={np.mean(losses):.6f}")

    # joint adversarial training
    model.init_ema()
    best_gap = float("inf")
    for epoch in range(JOINT_EPOCHS):
        model.train()
        g_losses, er_losses, d_losses = [], [], []
        p_real, p_fake = [], []

        for x in get_batches(train_loader, transform):
            losses = model.train_joint_step(x)
            g_losses.append(losses["generator"])
            er_losses.append(losses["embedder_recovery"])
            d_losses.append(losses["discriminator"])
            p_real.append(losses["d_real"])
            p_fake.append(losses["d_fake"])

        val, gap = run_validation(
            model, eval_loader, transform, with_generation=True
        )
        print(
            f"[Joint] {epoch + 1:03d} "
            f"G={np.mean(g_losses):.5f} "
            f"E/R={np.mean(er_losses):.5f} "
            f"D={np.mean(d_losses):.5f} "
            f"P(real|real)={np.mean(p_real):.2f} "
            f"P(real|fake)={np.mean(p_fake):.2f} "
            f"val_reconstruction={val:.5f} moment_gap={gap:.5f}"
            + " " + " ".join(f"val_event_{k}={v:.5f}" for k, v in model.last_validation_events.items())
        )

        # "best" = generated feature moments closest to real eval data.
        # Reconstruction MSE says nothing about generation quality.
        if gap < best_gap and (epoch + 1) >= MIN_BEST_EPOCH:
            best_gap = gap
            save_checkpoint(
                model,
                transform,
                CKPT_DIR / "timegan_best.pt",
                epoch=epoch + 1,
                moment_gap=gap,
            )
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                model,
                transform,
                CKPT_DIR / f"timegan_ep{epoch + 1:03d}.pt",
                epoch=epoch + 1,
                moment_gap=gap,
            )

    save_checkpoint(
        model,
        transform,
        CKPT_DIR / "timegan_last.pt",
        epoch=JOINT_EPOCHS,
    )
    print(f"Done. Best moment_gap={best_gap:.5f}")


if __name__ == "__main__":
    if RECONSTRUCTION_EVENT_WEIGHT < 0:
        raise ValueError("RECONSTRUCTION_EVENT_WEIGHT must be nonnegative")
    for name in ("AUTOENCODER_EPOCHS", "SUPERVISOR_EPOCHS", "JOINT_EPOCHS",
                 "STEPS_PER_EPOCH", "EVAL_WINDOWS"):
        if globals()[name] < 1:
            raise ValueError(f"{name} must be positive")
    MIN_BEST_EPOCH = min(MIN_BEST_EPOCH, JOINT_EPOCHS)
    train_set = LOBSTERDataset(
        root="LOBSTER_DATA",
        seq_len=SEQ_LEN,
        start_time=TRAIN_START,
        end_time=TRAIN_END,
    )
    eval_set = LOBSTERDataset(
        root="LOBSTER_DATA",
        seq_len=SEQ_LEN,
        start_time=EVAL_START,
        end_time=EVAL_END,
    )
    train_loader, eval_loader = make_loaders(train_set, eval_set)

    # Fit the scaler on the training period only (strided subset, CPU).
    transform = LOBFeatureTransform(tick_size=100.0)
    fit_loader = DataLoader(
        Subset(train_set, range(0, len(train_set), FIT_STRIDE)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    transform.fit(raw_feature_batches(fit_loader, transform))

    model = TimeGAN(
        input_dim=N_FEATURES,
        hidden_dim=128,
        noise_dim=64,
        num_layers=2,
        seq_len=SEQ_LEN,
        reconstruction_event_weight=RECONSTRUCTION_EVENT_WEIGHT,
    ).to(DEVICE)

    model.feature_mean.copy_(transform.mean.to(DEVICE))
    model.feature_std.copy_(transform.std.to(DEVICE))
    main(model, train_loader, eval_loader, transform)
