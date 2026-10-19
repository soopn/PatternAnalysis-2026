"""
Core Investigations:
    - does TimeGAN actually learn and enforce the structural constraints compared
    to standard baseline generator
    - does it reproduce authentic volatility clustering or noise/mode collapse

Recommendations:
    - compute KL divergence on spreads/returns (target KL <= 0.1)
    - evaluate depth heatmap structural similarity (price levels x time steps) (target SSIM > 0.6)
"""

from pathlib import Path
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from modules import TimeGAN
from dataset import LOBSTERDataset, LOBFeatureTransform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 100
BATCH_SIZE = 128
RAND_SEED = 69

# 09:30 -> 13:00
TRAIN_START, TRAIN_END = 34_200, 46_800

# 13:00 -> 14:30
EVAL_START, EVAL_END = 46_800, 52_200

# 14:30 -> 16:00
TEST_START, TEST_END = 52_200, 57_600

AUTOENCODER_EPOCHS = 100
SUPERVISOR_EPOCHS = 100
JOINT_EPOCHS = 200

random.seed(RAND_SEED)
np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RAND_SEED)


def make_all_datasets():
    train_dataset = LOBSTERDataset(
        root="LOBSTER_DATA",
        seq_len=100,
        start_time=TRAIN_START,
        end_time=TRAIN_END,
    )
    eval_dataset = LOBSTERDataset(
        root="LOBSTER_DATA",
        seq_len=100,
        start_time=EVAL_START,
        end_time=EVAL_END,
    )
    test_dataset = LOBSTERDataset(
        root="LOBSTER_DATA",
        seq_len=100,
        start_time=TEST_START,
        end_time=TEST_END,
    )

    return train_dataset, eval_dataset, test_dataset


def make_all_loaders(train_set, eval_set, test_set):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return train_loader, eval_loader, test_loader


def get_features(batch):
    raw = feature_transform.transform_raw(batch["orderbook"])
    return feature_transform.normalise(raw)


def get_batches(loader):
    for batch in loader:
        x = get_features(batch)
        yield x.to(DEVICE)


def raw_feature_batches(loader):
    for batch in loader:
        yield feature_transform.transform_raw(batch["orderbook"])


def run_validation(model, loader):
    """
    Evaluate reconstruction on real validation sequences.
    Assumes model.reconstruct(x) returns reconstructed features.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x in get_batches(loader):
            x_hat = model.reconstruct(x)

            # Weight batches by their number of sequences.
            loss = torch.nn.functional.mse_loss(x_hat, x, reduction="mean")

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    model.train()

    return total_loss / max(total_samples, 1)


def main(model, train_loader, eval_loader, feature_dim=40):
    """Wrapper to run the actual model training script"""
    Path("checkpoints").mkdir(exist_ok=True)

    # best_val_loss = float("inf")

    # embedder + recovery autoencoder
    for epoch in range(AUTOENCODER_EPOCHS):
        model.train()
        losses = []

        for x in get_batches(train_loader):
            loss = model.pretrain_autoencoder(x)
            losses.append(float(loss))

        eval_loss = run_validation(model, eval_loader)

        print(
            f"[Autoencoder] {epoch + 1:03d} "
            f"train={np.mean(losses):.6f} "
            f"val_reconstruction={eval_loss:.6f}"
        )

    # supervisor learns latent temporal dynamics
    for epoch in range(SUPERVISOR_EPOCHS):
        model.train()
        losses = []

        for x in get_batches(train_loader):
            loss = model.pretrain_supervisor(x)
            losses.append(float(loss))

        print(f"[Supervisor] {epoch + 1:03d} " f"train={np.mean(losses):.6f}")

    # joint adversarial training
    for epoch in range(JOINT_EPOCHS):
        model.train()
        g_losses = []
        er_losses = []
        d_losses = []

        for x in get_batches(train_loader):
            losses = model.train_joint_step(x)

            g_losses.append(losses["generator"])
            er_losses.append(losses["embedder_recovery"])
            d_losses.append(losses["discriminator"])

        eval_loss = run_validation(model, eval_loader)

        print(
            f"[Joint] {epoch + 1:03d} "
            f"G={np.mean(g_losses):.5f} "
            f"E/R={np.mean(er_losses):.5f} "
            f"D={np.mean(d_losses):.5f} "
            f"val_reconstruction={eval_loss:.5f}"
        )

    # save model after training
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_transform.mean,
            "feature_std": feature_transform.std,
        },
        "checkpoints/timegan_best.pt",
    )


if __name__ == "__main__":
    train_dataset, eval_dataset, test_dataset = make_all_datasets()
    train_loader, eval_loader, test_loader = make_all_loaders(
        train_dataset, eval_dataset, test_dataset
    )

    model = TimeGAN(
        input_dim=40,
        hidden_dim=128,
        noise_dim=64,
        num_layers=2,
        seq_len=SEQ_LEN,
    ).to(DEVICE)

    # Fit on the training period only.
    feature_transform = LOBFeatureTransform(tick_size=100.0)
    feature_transform.fit(raw_feature_batches(train_loader))
    batch = next(iter(train_loader))
    raw = batch["orderbook"]

    features = feature_transform.transform_raw(raw)
    normalised = feature_transform.normalise(features)

    feature_dim = features.shape[2]

    main(train_loader, eval_loader, feature_dim)
