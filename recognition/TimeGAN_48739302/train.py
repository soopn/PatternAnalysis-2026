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
import time

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
CHECKPOINT_EVERY = 25  # joint epochs

CKPT_DIR = Path("checkpoints")

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RAND_SEED)


def strided_subset(dataset, max_items):
    """Evenly spaced windows across the whole period."""
    stride = max(1, len(dataset) // max_items)
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
    """Mean |difference| of per-timestep, per-feature means and stds
    (standardised units), same statistics as the generator's moment loss."""
    d_mean = (real.mean(0) - fake.mean(0)).abs().mean()
    d_std = (real.std(0) - fake.std(0)).abs().mean()
    return (d_mean + d_std).item()


@torch.no_grad()
def run_validation(model, loader, transform, with_generation=False):
    model.eval()

    total, n = 0.0, 0
    real = []
    for x in get_batches(loader, transform):
        x_hat = model.reconstruct(x)
        total += F.mse_loss(x_hat, x).item() * x.size(0)
        n += x.size(0)
        if with_generation:
            real.append(x)

    gap = None
    if with_generation and real:
        real = torch.cat(real)
        fake = model.generate(real.size(0), real.size(1))
        gap = moment_gap(real, fake)

    model.train()
    return total / max(n, 1), gap


def save_checkpoint(model, transform, path, **extra):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": transform.mean,
            "feature_std": transform.std,
            **extra,
        },
        path,
    )


def main(model, train_loader, eval_loader, transform):
    CKPT_DIR.mkdir(exist_ok=True)

    # embedder + recoverer autoencoder
    encoder_start = time.time()
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
        )
    encoder_end = time.time()

    # supervisor learns latent temporal dynamics
    supervisor_start = time.time()
    for epoch in range(SUPERVISOR_EPOCHS):
        model.train()
        losses = [
            model.pretrain_supervisor(x)
            for x in get_batches(train_loader, transform)
        ]
        print(f"[Supervisor] {epoch + 1:03d} train={np.mean(losses):.6f}")
    supervisor_end = time.time()

    # joint adversarial training
    joint_start = time.time()
    best_gap = float("inf")
    for epoch in range(JOINT_EPOCHS):
        model.train()
        g_losses, er_losses, d_losses = [], [], []

        for x in get_batches(train_loader, transform):
            losses = model.train_joint_step(x)
            g_losses.append(losses["generator"])
            er_losses.append(losses["embedder_recovery"])
            d_losses.append(losses["discriminator"])

        val, gap = run_validation(
            model, eval_loader, transform, with_generation=True
        )
        print(
            f"[Joint] {epoch + 1:03d} "
            f"G={np.mean(g_losses):.5f} "
            f"E/R={np.mean(er_losses):.5f} "
            f"D={np.mean(d_losses):.5f} "
            f"val_reconstruction={val:.5f} moment_gap={gap:.5f}"
        )

        # "best" = generated feature moments closest to real eval data.
        # Reconstruction MSE says nothing about generation quality.
        if gap < best_gap:
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
    joint_end = time.time()

    save_checkpoint(
        model,
        transform,
        CKPT_DIR / "timegan_last.pt",
        epoch=JOINT_EPOCHS,
    )
    print(f"Done. Best moment_gap={best_gap:.5f}")
    print(
        "Training Durations:\n"
        + f"  Autoencoder: {encoder_end-encoder_start}s\n"
        + f"  Supervisor: {supervisor_end-supervisor_start}s\n"
        + f"  Joint: {joint_end-joint_start}s\n"
    )


if __name__ == "__main__":
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
        # TODO: try bidirectional discriminator
        # bidirec_discrim=True,
    ).to(DEVICE)

    main(model, train_loader, eval_loader, transform)
