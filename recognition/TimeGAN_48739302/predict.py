from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from pandas import DataFrame

from dataset import LOBFeatureTransform, LOBSTERDataset
from modules import TimeGAN

SEQ_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_START, TEST_END = 52_200, 57_600
BATCH_SIZE = 128


def raw_feature_batches(loader):
    for batch in loader:
        yield feature_transform.transform_raw(batch["orderbook"])


@torch.no_grad()
def generate_sequences(
    model,
    n_samples=100,
    seq_len=100,
    noise_dim=64,
    device="cpu",
):
    model.eval()

    z = torch.randn(
        n_samples,
        seq_len,
        noise_dim,
        device=device,
    )

    generated_latent = model.generator(z)
    supervised_latent = model.supervisor(generated_latent)
    generated_features = model.recoverer(supervised_latent)

    return generated_features.cpu()


def plot_generated_lob(
    generated,
    feature_transform,
    sample_idx=0,
    timestep=0,
):
    # Select one generated order-book state.
    x = generated[sample_idx, timestep]

    # Undo feature standardisation.
    x = feature_transform.inverse_normalise(x)

    # Each level contains:
    # [ask price offset, log ask volume, bid price offset, log bid volume]
    levels = x.reshape(10, 4)

    ask_offset = levels[:, 0]
    ask_volume = np.expm1(levels[:, 1])

    bid_offset = levels[:, 2]
    bid_volume = np.expm1(levels[:, 3])

    ask_valid = np.isfinite(ask_offset) & np.isfinite(ask_volume)
    bid_valid = np.isfinite(bid_offset) & np.isfinite(bid_volume)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        ask_offset[ask_valid],
        ask_volume[ask_valid],
        height=0.3,
        label="Ask",
        alpha=0.7,
    )
    ax.barh(
        bid_offset[bid_valid],
        bid_volume[bid_valid],
        height=0.3,
        label="Bid",
        alpha=0.7,
    )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Price offset from midprice (ticks)")
    ax.set_ylabel("Price level")
    ax.set_title(f"Generated LOB: sample {sample_idx}, t={timestep}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figs/generated_lob.png")


# TODO: generate an actual message and orderbook csv's into out/
if __name__ == "__main__":
    Path("out").mkdir(exist_ok=True)

    test_dataset = LOBSTERDataset(
        root="LOBSTER_DATA",
        seq_len=100,
        start_time=TEST_START,
        end_time=TEST_END,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )
    feature_transform = LOBFeatureTransform(tick_size=100.0)
    feature_transform.fit(raw_feature_batches(test_loader))

    model = TimeGAN(
        input_dim=40,
        hidden_dim=128,
        noise_dim=64,
        num_layers=2,
        seq_len=SEQ_LEN,
    ).to(DEVICE)

    checkpoint = torch.load(
        "checkpoints/timegan_best.pt",
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    feature_transform.mean = checkpoint["feature_mean"]
    feature_transform.std = checkpoint["feature_std"]

    with torch.no_grad():
        generated = model.generate(
            BATCH_SIZE,
            seq_len=100,
        )
        gen_lob = generate_sequences(
            model, n_samples=1000, seq_len=100, noise_dim=64, device=DEVICE
        )
        features = feature_transform.inverse_normalise(gen_lob)

    plot_generated_lob(generated.cpu(), feature_transform)
    raw_lob = feature_transform.inverse_normalise(features)

    # save to csv
    rows = raw_lob.reshape(-1, 40)
    DataFrame(rows).to_csv("out/generated.csv", index=False, header=False)
