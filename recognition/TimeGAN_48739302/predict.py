"""Generate synthetic LOBSTER Level-10 order books from a trained TimeGAN.

Writes:
    out/generated_features.npy   (N, T, 42) features in original units
    out/generated_raw_lob.npy    (N, T, 40) LOBSTER-style rows
        [ask_price, ask_vol, bid_price, bid_vol] x 10
    out/generated_raw_lob.csv    long format: sample, t, time_s, 40 columns
    figs/generated_lob.png       depth plot of one generated snapshot

Decoding guarantees (by construction, not by learning):
    - spread >= 1 tick, so no crossed/locked books
    - prices strictly increase (asks) / decrease (bids) with depth
    - all prices lie on the tick grid
The raw-feature violation rates printed by structure_report() show how much
of that structure the network learned on its own.

Midprice paths are the cumulative sum of the generated midprice changes,
anchored at the first midprice of random real test windows.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import DataFrame

from dataset import (
    DT_FEATURE,
    DT_SCALE,
    MID_FEATURE,
    N_FEATURES,
    LOBFeatureTransform,
    LOBSTERDataset,
)
import sparsify
from modules import TimeGAN

SEQ_LEN = 100
N_SAMPLES = 1000
TICK_SIZE = 100.0  # $0.01 * 1e4
TEST_START, TEST_END = 52_200, 57_600
CHECKPOINT = "checkpoints/timegan_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 123
EMPTY_VOL = 0.5  # volumes are integers >= 1, so < 0.5 means "empty level"
PERIODS = {
    "train": (34_200, 46_800),
    "eval": (46_800, 52_200),
    "test": (TEST_START, TEST_END),
}
BURN_IN = 20  # extra leading steps generated and dropped (GRU warm-up transient)
SPARSE_PARAMS = None  # set by enable_sparsify(); see sparsify.py


def real_period():
    """(start, end) of the real data to compare with; REAL_PERIOD=train|eval|test."""
    return PERIODS[os.environ.get("REAL_PERIOD", "test")]


def load_model(path=CHECKPOINT):
    global SPARSE_PARAMS
    SPARSE_PARAMS = None
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    config = checkpoint.get("model_config", {
        "input_dim": N_FEATURES, "hidden_dim": 128, "noise_dim": 64,
        "num_layers": 2, "seq_len": SEQ_LEN,
    })
    model = TimeGAN(**config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = LOBFeatureTransform(tick_size=TICK_SIZE)
    transform.mean = checkpoint["feature_mean"].cpu()
    transform.std = checkpoint["feature_std"].cpu()
    model.feature_mean.copy_(transform.mean.to(DEVICE))
    model.feature_std.copy_(transform.std.to(DEVICE))
    if os.environ.get("SPARSIFY") == "1":  # e.g. SPARSIFY=1 python evaluate.py
        enable_sparsify(model, transform)
    return model, transform


def enable_sparsify(model, transform, n=2000):
    """Calibrate the hold filter against real TRAINING-period windows so every
    later generate_sequences() call returns event-sparse features."""
    global SPARSE_PARAMS
    ds = LOBSTERDataset(
        root="LOBSTER_DATA", seq_len=SEQ_LEN, start_time=34_200, end_time=46_800
    )
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    ob = torch.stack([ds[int(i)]["orderbook"] for i in idx])
    ms = torch.stack([ds[int(i)]["message"] for i in idx])
    real = transform.transform_raw(ob, ms).numpy()
    devices = [DEVICE.index or 0] if DEVICE.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(SEED + 1)
        raw = generate_sequences(model, transform, n, SEQ_LEN, sparse=False)
    SPARSE_PARAMS = sparsify.calibrate(raw, real)
    after = sparsify.apply(raw, SPARSE_PARAMS, seed=1)
    for name, f in (("real", real), ("generated", raw), ("sparsified", after)):
        lat = sparsify.lattice_stats(f)
        print(
            f"sparsify {name:<10s} unchanged {sparsify.unchanged_share(f).mean():.3f}  "
            f"mult100 {lat['mult100']:.3f}  mult10 {lat['mult10']:.3f}  "
            f"1-tick gaps {lat['gap1']:.3f}"
        )


@torch.no_grad()
def generate_sequences(model, transform, n_samples=100, seq_len=SEQ_LEN,
                       burn_in=BURN_IN, sparse=True):
    """Sample with the model's own noise; return original-unit features,
    shape (N, T, 42), numpy. The generator's GRU starts from a zero state and
    the first ~10 steps are rough (spread roughness 1.3 vs 0.2 for real), so
    burn_in extra steps are generated and dropped."""
    model.eval()
    chunks = []
    for start in range(0, n_samples, 256):
        b = min(256, n_samples - start)
        chunks.append(model.generate(b, seq_len + burn_in)[:, burn_in:].cpu())
    generated = torch.cat(chunks)  # standardised space
    features = transform.inverse_normalise(generated).numpy()
    if sparse and SPARSE_PARAMS is not None:
        features = sparsify.apply(features, SPARSE_PARAMS, seed=SEED)
    return features


def real_start_mids(n_samples, rng):
    """First midprice (in ticks) of random real test windows."""
    ds = LOBSTERDataset(
        root="LOBSTER_DATA",
        seq_len=SEQ_LEN,
        start_time=TEST_START,
        end_time=TEST_END,
    )
    idx = rng.choice(len(ds), size=n_samples, replace=len(ds) < n_samples)
    mids = []
    for i in idx:
        ob = ds[int(i)]["orderbook"].numpy()
        mids.append(0.5 * (float(ob[0, 0]) + float(ob[0, 2])) / TICK_SIZE)
    return np.asarray(mids, dtype=np.float64)


def structure_report(features):
    """Violation rates in the RAW generated features (before decoding)."""
    lv = features[..., :40].reshape(-1, 10, 4).astype(np.float64)
    ask_gap, bid_gap = lv[..., 0], lv[..., 2]
    ask_ne = np.expm1(np.maximum(lv[..., 1], 0)) >= EMPTY_VOL
    bid_ne = np.expm1(np.maximum(lv[..., 3], 0)) >= EMPTY_VOL

    spread = ask_gap[:, 0] + bid_gap[:, 0]
    tight = spread < 0.5  # rounds to < 1 tick
    sub_tick = (
        ((ask_gap[:, 1:] < 0.5) & ask_ne[:, 1:]).any(axis=1)
        | ((bid_gap[:, 1:] < 0.5) & bid_ne[:, 1:]).any(axis=1)
    )
    neg_vol = (lv[..., 1] < 0).any(axis=1) | (lv[..., 3] < 0).any(axis=1)
    empty_l1 = ~ask_ne[:, 0] | ~bid_ne[:, 0]

    print("Constraint violations in raw generated features:")
    print(f"  spread < 1 tick (crossed/locked):       {tight.mean():.2%}")
    print(f"  sub-tick gap at a non-empty deep level: {sub_tick.mean():.2%}")
    print(f"  negative log-volumes:                   {neg_vol.mean():.2%}")
    print(f"  empty level 1 (either side):            {empty_l1.mean():.2%}")
    if (~tight).any():
        print(
            f"  spread (ticks): mean={spread[~tight].mean():.2f} "
            f"median={np.median(spread[~tight]):.2f}"
        )


def decode_features(features, start_mid_ticks):
    """(N, T, 42) original-unit features -> structurally valid book.

    start_mid_ticks: (N,) midprice at t=0 in ticks.

    Returns a dict of arrays (prices in ticks, volumes in shares):
        ask_price, ask_vol, bid_price, bid_vol : (N, T, 10)
        mid, dt, time                          : (N, T)
    """
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 3 or f.shape[-1] != N_FEATURES:
        raise ValueError(f"features must have shape (N, T, {N_FEATURES})")
    start_mid_ticks = np.asarray(start_mid_ticks, dtype=np.float64)
    n, t, _ = f.shape
    if start_mid_ticks.shape != (n,):
        raise ValueError("start_mid_ticks must have shape (N,)")

    lv = f[..., :40].reshape(n, t, 10, 4)
    ask_gap, bid_gap = lv[..., 0], lv[..., 2]
    ask_vol = np.rint(np.expm1(np.maximum(lv[..., 1], 0.0)))
    bid_vol = np.rint(np.expm1(np.maximum(lv[..., 3], 0.0)))
    ask_vol[ask_vol < EMPTY_VOL] = 0
    bid_vol[bid_vol < EMPTY_VOL] = 0

    # Midprice path on the half-tick grid.
    d_mid = np.rint(np.sinh(f[..., MID_FEATURE]) * 2.0) / 2.0  # undo asinh
    d_mid[:, 0] = 0.0
    mid = start_mid_ticks[:, None] + np.cumsum(d_mid, axis=1)

    # Level 1: integer spread >= 1 tick centred on the mid.
    spread = np.maximum(1.0, np.rint(ask_gap[..., 0] + bid_gap[..., 0]))
    best_bid = np.rint(mid - spread / 2.0)
    best_ask = best_bid + spread

    # Deeper levels: each step is >= 1 tick.
    ask_steps = np.maximum(1.0, np.rint(ask_gap[..., 1:]))
    bid_steps = np.maximum(1.0, np.rint(bid_gap[..., 1:]))
    zeros = np.zeros((n, t, 1))
    ask_price = best_ask[..., None] + np.concatenate(
        [zeros, np.cumsum(ask_steps, axis=-1)], axis=-1
    )
    bid_price = best_bid[..., None] - np.concatenate(
        [zeros, np.cumsum(bid_steps, axis=-1)], axis=-1
    )

    dt = np.expm1(np.maximum(f[..., DT_FEATURE], 0.0)) / DT_SCALE
    time = np.cumsum(dt, axis=1) - dt[:, :1]

    return {
        "ask_price": ask_price,
        "ask_vol": ask_vol,
        "bid_price": bid_price,
        "bid_vol": bid_vol,
        "mid": mid,
        "dt": dt,
        "time": time,
    }


def to_lobster_rows(dec, tick_size=TICK_SIZE):
    """Decoded book -> LOBSTER rows (N, T, 40) with sentinel prices on
    empty levels."""
    ask_price = np.where(
        dec["ask_vol"] > 0, dec["ask_price"] * tick_size, 9_999_999_999
    )
    bid_price = np.where(
        dec["bid_vol"] > 0, dec["bid_price"] * tick_size, -9_999_999_999
    )
    raw = np.stack(
        [ask_price, dec["ask_vol"], bid_price, dec["bid_vol"]], axis=-1
    )
    n, t = raw.shape[:2]
    return raw.reshape(n, t, 40)


def plot_generated_lob(dec, sample_idx=0, timestep=0, path="figs/generated_lob.png"):
    mid = dec["mid"][sample_idx, timestep]
    ask_off = dec["ask_price"][sample_idx, timestep] - mid
    bid_off = dec["bid_price"][sample_idx, timestep] - mid  # negative
    ask_vol = dec["ask_vol"][sample_idx, timestep]
    bid_vol = dec["bid_vol"][sample_idx, timestep]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ask_off[ask_vol > 0], ask_vol[ask_vol > 0], height=0.3,
            label="Ask", alpha=0.7)
    ax.barh(bid_off[bid_vol > 0], bid_vol[bid_vol > 0], height=0.3,
            label="Bid", alpha=0.7)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Volume (shares)")
    ax.set_ylabel("Price offset from midprice (ticks)")
    ax.set_title(f"Generated LOB: sample {sample_idx}, t={timestep}")
    ax.legend()
    fig.tight_layout()

    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default=CHECKPOINT)
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    n_samples = args.samples
    Path("out").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    model, transform = load_model(args.checkpoint)

    features = generate_sequences(model, transform, n_samples, SEQ_LEN)
    np.save("out/generated_features.npy", features)
    structure_report(features)

    dec = decode_features(features, real_start_mids(n_samples, rng))
    raw = to_lobster_rows(dec)
    np.save("out/generated_raw_lob.npy", raw)

    cols = [
        f"{side}_{kind}_{lvl}"
        for lvl in range(1, 11)
        for side, kind in (
            ("ask", "price"),
            ("ask", "volume"),
            ("bid", "price"),
            ("bid", "volume"),
        )
    ]
    df = DataFrame(raw.reshape(-1, 40), columns=None)
    df.to_csv("out/generated_raw_lob.csv", index=False)

    plot_generated_lob(dec)
    print("Saved to out/ and figs/")


if __name__ == "__main__":
    main()
