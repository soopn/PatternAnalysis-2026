"""Evaluate generated order books against real test-period windows.

Metrics
    KL divergence  KL(real || generated) on histograms of
                   spread (ticks), 1-step and 10-step midprice change (ticks),
                   log inter-event time.                    target: <= 0.1
    SSIM           structural similarity of depth heatmaps
                   (price levels x time steps).             target: > 0.6
                   Reported for generated-vs-real pairs, a real-vs-real
                   baseline (an unpaired reference) and for the mean heatmaps.
    Volatility     autocorrelation of |midprice change| at lags 1..10,
    clustering     real vs generated.

Requires:  pip install scikit-image
Writes:    out/metrics.json, figs/depth_heatmaps.png, figs/abs_return_acf.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

from dataset import DT_COL, DT_SCALE, LOBSTERDataset
from predict import (
    SEED,
    SEQ_LEN,
    TEST_END,
    TEST_START,
    TICK_SIZE,
    decode_features,
    generate_sequences,
    load_model,
    to_lobster_rows,
)

N_EVAL = 500
N_SSIM_PAIRS = 300
KL_TARGET = 0.1
SSIM_TARGET = 0.6
HALF_RANGE_TICKS = 15  # price heatmap covers mid +/- 15 ticks
MAX_LAG = 10


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
PERIODS = {
    "train": (34_200, 46_800),  # 09:30-13:00, what the generator is trained on
    "eval": (46_800, 52_200),  # 13:00-14:30
    "test": (TEST_START, TEST_END),  # 14:30-16:00
}


def load_real_windows(n, start=TEST_START, end=TEST_END):
    """Evenly spaced real windows -> raw rows (N, T, 40), dt (N, T)."""
    ds = LOBSTERDataset(
        root="LOBSTER_DATA",
        seq_len=SEQ_LEN,
        start_time=start,
        end_time=end,
    )
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    ob = np.stack([ds[int(i)]["orderbook"].numpy() for i in idx])
    dt = np.stack([ds[int(i)]["message"][:, DT_COL].numpy() for i in idx])
    return ob.astype(np.float64), dt.astype(np.float64)


# --------------------------------------------------------------------------
# summaries computed identically for real and generated data
# --------------------------------------------------------------------------
def summarise(raw, dt):
    n, t, _ = raw.shape
    lv = raw.reshape(n, t, 10, 4)
    a1, b1 = lv[:, :, 0, 0], lv[:, :, 0, 2]
    valid = (
        (lv[:, :, 0, 1] > 0)
        & (lv[:, :, 0, 3] > 0)
        & (np.abs(a1) < 1e9)
        & (np.abs(b1) < 1e9)
    )

    spread = ((a1 - b1) / TICK_SIZE)[valid]
    mid = np.where(valid, 0.5 * (a1 + b1) / TICK_SIZE, np.nan)
    d1 = np.diff(mid, axis=1)  # (N, T-1), NaN where a level 1 is empty
    r10 = mid[:, 10:] - mid[:, :-10]
    log_dt = np.log1p(dt[:, 1:] * DT_SCALE)  # first dt is not generated

    return {
        "spread": spread,
        "d1": d1,
        "ret1": d1[np.isfinite(d1)],
        "ret10": r10[np.isfinite(r10)],
        "logret": np.diff(np.log(np.where(valid, mid, np.nan)), axis=1)[
            np.isfinite(d1)
        ],
        "log_dt": log_dt.reshape(-1),
    }


def abs_return_acf(d1, max_lag=MAX_LAG):
    """Pooled within-window autocorrelation of |midprice change|."""
    a = np.abs(d1)
    mu, var = np.nanmean(a), np.nanvar(a)
    if not np.isfinite(var) or var == 0:
        return np.zeros(max_lag)
    out = []
    for k in range(1, max_lag + 1):
        x, y = a[:, :-k], a[:, k:]
        m = np.isfinite(x) & np.isfinite(y)
        out.append(np.mean((x[m] - mu) * (y[m] - mu)) / var if m.any() else np.nan)
    return np.asarray(out)


# --------------------------------------------------------------------------
# KL divergence
# --------------------------------------------------------------------------
def kl_divergence(real, fake, edges, eps=1e-6):
    """KL(real || fake) on a shared histogram. Values outside the edges are
    clipped into the outermost bins. Penalises modes the generator misses."""
    if edges is None or len(real) == 0 or len(fake) == 0:
        return float("nan")
    p, _ = np.histogram(np.clip(real, edges[0], edges[-1]), bins=edges)
    q, _ = np.histogram(np.clip(fake, edges[0], edges[-1]), bins=edges)
    p = p / p.sum() + eps
    q = q / q.sum() + eps
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def kl_repo_style(real, fake, bins=100, eps=1e-12):
    """KL(real || fake) built like the other repo's kl_divergence_hist:
    100 uniform bins over the union range of both samples (range always
    includes 0), smoothing eps=1e-12. Reported for comparability only."""
    if len(real) == 0 or len(fake) == 0:
        return float("nan")
    lo = min(real.min(initial=0.0), fake.min(initial=0.0))
    hi = max(real.max(initial=0.0), fake.max(initial=0.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    r, edges = np.histogram(real, bins=bins, range=(lo, hi))
    f, _ = np.histogram(fake, bins=edges)
    p = r.astype(np.float64) + eps
    q = f.astype(np.float64) + eps
    p, q = p / p.sum(), q / q.sum()
    return float(max(np.sum(p * (np.log(p) - np.log(q))), 0.0))


def kl_repo_report(real, fake):
    return {
        "spread_ticks": kl_repo_style(real["spread"], fake["spread"]),
        "log_return_1step": kl_repo_style(real["logret"], fake["logret"]),
    }


def quantile_edges(real, n_bins=20):
    if len(real) == 0:
        return None
    edges = np.unique(np.quantile(real, np.linspace(0, 1, n_bins + 1)))
    return edges if len(edges) > 2 else None


def kl_report(real, fake):
    specs = {
        "spread_ticks": ("spread", quantile_edges(real["spread"], 30)),
        "return_1step_ticks": ("ret1", np.arange(-5.25, 5.30, 0.5)),
        "return_10step_ticks": ("ret10", quantile_edges(real["ret10"])),
        "log_inter_event_time": ("log_dt", quantile_edges(real["log_dt"])),
    }
    return {
        name: kl_divergence(real[key], fake[key], edges)
        for name, (key, edges) in specs.items()
    }


# --------------------------------------------------------------------------
# depth heatmaps + SSIM
# --------------------------------------------------------------------------
def heatmap_levels(raw):
    """(N, 20, T): log1p volume per book level; rows are ask L10..L1 then
    bid L1..L10."""
    n, t, _ = raw.shape
    lv = raw.reshape(n, t, 10, 4)
    ask = np.log1p(np.maximum(lv[..., 1], 0))
    bid = np.log1p(np.maximum(lv[..., 3], 0))
    img = np.concatenate([ask[..., ::-1], bid], axis=-1)  # (N, T, 20)
    return img.transpose(0, 2, 1)


def heatmap_prices(raw, half_range=HALF_RANGE_TICKS):
    """(N, 2R, T): log1p volume at 1-tick price bins around the midprice."""
    n, t, _ = raw.shape
    lv = raw.reshape(n, t, 10, 4)
    mid = 0.5 * (lv[:, :, 0, 0] + lv[:, :, 0, 2])
    img = np.zeros((n, 2 * half_range, t))
    n_idx = np.broadcast_to(np.arange(n)[:, None, None], (n, t, 10))
    t_idx = np.broadcast_to(np.arange(t)[None, :, None], (n, t, 10))
    for price_col, vol_col in ((0, 1), (2, 3)):
        off = (lv[..., price_col] - mid[..., None]) / TICK_SIZE
        vol = lv[..., vol_col]
        with np.errstate(invalid="ignore"):
            rows = np.floor(off).astype(np.int64) + half_range
        ok = (
            (vol > 0)
            & np.isfinite(off)
            & (rows >= 0)
            & (rows < 2 * half_range)
        )
        np.add.at(img, (n_idx[ok], rows[ok], t_idx[ok]), vol[ok])
    return np.log1p(img)


def mean_ssim(a, b, data_range):
    return float(
        np.mean(
            [
                structural_similarity(x, y, data_range=data_range)
                for x, y in zip(a, b)
            ]
        )
    )


def ssim_report(real_maps, fake_maps, rng):
    data_range = float(real_maps.max() - real_maps.min()) or 1.0
    k = min(N_SSIM_PAIRS, len(real_maps), len(fake_maps))

    perm_r = rng.permutation(len(real_maps))[:k]
    perm_f = rng.permutation(len(fake_maps))[:k]
    gen_vs_real = mean_ssim(real_maps[perm_r], fake_maps[perm_f], data_range)

    # Different real windows against each other: what "perfect" scores.
    real_vs_real = mean_ssim(
        real_maps[perm_r], real_maps[np.roll(perm_r, 1)], data_range
    )

    mean_maps = structural_similarity(
        real_maps.mean(axis=0), fake_maps.mean(axis=0), data_range=data_range
    )
    return {
        "generated_vs_real": gen_vs_real,
        "real_vs_real_baseline": real_vs_real,
        "mean_heatmaps": float(mean_maps),
    }


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def plot_heatmaps(real_maps, fake_maps, path="figs/depth_heatmaps.png"):
    vmax = max(real_maps.mean(0).max(), fake_maps.mean(0).max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, maps, title in zip(
        axes, (real_maps, fake_maps), ("Real (mean)", "Generated (mean)")
    ):
        im = ax.imshow(maps.mean(0), aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("time step")
    axes[0].set_ylabel("ask L10..L1 | bid L1..L10")
    fig.colorbar(im, ax=axes, label="log1p(volume)")
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_acf(acf_real, acf_fake, path="figs/abs_return_acf.png"):
    lags = np.arange(1, len(acf_real) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lags, acf_real, "o-", label="Real")
    ax.plot(lags, acf_fake, "s-", label="Generated")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("lag (events)")
    ax.set_ylabel("ACF of |midprice change|")
    ax.set_title("Volatility clustering")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    Path("out").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    real_raw, real_dt = load_real_windows(N_EVAL)

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    model, transform = load_model(ckpt) if ckpt else load_model()
    features = generate_sequences(model, transform, N_EVAL, SEQ_LEN)
    start_mid = 0.5 * (real_raw[:, 0, 0] + real_raw[:, 0, 2]) / TICK_SIZE
    dec = decode_features(features, start_mid)
    fake_raw = to_lobster_rows(dec)

    real_s = summarise(real_raw, real_dt)
    fake_s = summarise(fake_raw, dec["dt"])

    kl = kl_report(real_s, fake_s)

    # Noise floor: KL between two interleaved halves of the REAL windows.
    # Shows how small KL can realistically get with this binning and sample size.
    half_a = summarise(real_raw[0::2], real_dt[0::2])
    half_b = summarise(real_raw[1::2], real_dt[1::2])
    kl_floor = kl_report(half_a, half_b)
    kl_repo = kl_repo_report(real_s, fake_s)
    kl_repo_floor = kl_repo_report(half_a, half_b)
    ssim = {
        "levels": ssim_report(
            heatmap_levels(real_raw), heatmap_levels(fake_raw), rng
        ),
        "prices": ssim_report(
            heatmap_prices(real_raw), heatmap_prices(fake_raw), rng
        ),
    }
    acf_real = abs_return_acf(real_s["d1"])
    acf_fake = abs_return_acf(fake_s["d1"])

    q = [5, 25, 50, 75, 95]
    print("spread quantiles (ticks) %s" % q)
    print("  real:     ", np.round(np.percentile(real_s["spread"], q), 1))
    print("  generated:", np.round(np.percentile(fake_s["spread"], q), 1))
    for key, label in (("ret10", "10-step midprice change (ticks)"),
                       ("log_dt", "log inter-event time")):
        print(f"{label} quantiles {q}")
        print("  real:     ", np.round(np.percentile(real_s[key], q), 2))
        print("  generated:", np.round(np.percentile(fake_s[key], q), 2))

    print("\nKL(real || generated)       target <= %.2f" % KL_TARGET)
    print(f"  {'':<24s} {'generated':>9s}   real-vs-real floor")
    for name, v in kl.items():
        flag = "PASS" if v <= KL_TARGET else "FAIL"
        print(f"  {name:<24s} {v:9.4f}  {flag}   {kl_floor[name]:8.4f}")

    # Is the mismatch model error or time-of-day drift? The generator can only
    # learn the training period; the metrics above compare with the test period.
    period_s = {}
    for pname, (a, b) in PERIODS.items():
        p_raw, p_dt = load_real_windows(N_EVAL, a, b)
        period_s[pname] = summarise(p_raw, p_dt)
    print("\nSpread (ticks) quantiles %s by period" % q)
    for pname, s_p in period_s.items():
        print(f"  real {pname:<6s}", np.round(np.percentile(s_p["spread"], q), 1))
    print("  generated   ", np.round(np.percentile(fake_s["spread"], q), 1))
    kl_periods = {
        pname: kl_report(s_p, fake_s) for pname, s_p in period_s.items()
    }
    kl_shift = {
        pname: kl_report(period_s["train"], s_p)
        for pname, s_p in period_s.items()
        if pname != "train"
    }
    print("\nKL(real period || generated), and real train -> real period shift")
    print(f"  {'':<22s} {'vs train':>9s} {'vs eval':>9s} {'vs test':>9s}"
          f" {'shift train>eval':>17s} {'shift train>test':>17s}")
    for name in kl:
        print(f"  {name:<22s} {kl_periods['train'][name]:9.4f} "
              f"{kl_periods['eval'][name]:9.4f} {kl_periods['test'][name]:9.4f}"
              f" {kl_shift['eval'][name]:17.4f} {kl_shift['test'][name]:17.4f}")
    print("  If 'vs train' is small while 'vs test' is large and matches the")
    print("  shift column, the generator reproduces the training period and the")
    print("  test KL mostly measures time-of-day drift.")

    print("\nKL, repo-style (100 uniform bins over union range, eps=1e-12)")
    print(f"  {'':<24s} {'generated':>9s}   real-vs-real floor")
    for name, v in kl_repo.items():
        print(f"  {name:<24s} {v:9.4f}  {'PASS' if v <= KL_TARGET else 'FAIL'}   "
              f"{kl_repo_floor[name]:8.4f}")

    print("\nSSIM of depth heatmaps      target > %.2f" % SSIM_TARGET)
    for kind, r in ssim.items():
        print(f"  [{kind}-indexed heatmap]")
        for name, v in r.items():
            print(f"    {name:<24s} {v:8.4f}")
    print("  Compare generated_vs_real with real_vs_real_baseline: unpaired")
    print("  windows rarely match exactly; this baseline is a reference, not a ceiling.")

    print("\n|midprice change| autocorrelation (lag 1..%d)" % MAX_LAG)
    print("  real:     ", np.round(acf_real, 3))
    print("  generated:", np.round(acf_fake, 3))

    with open("out/metrics.json", "w") as fh:
        json.dump(
            {
                "kl": kl,
                "kl_real_vs_real_floor": kl_floor,
                "kl_by_period": kl_periods,
                "kl_period_shift_from_train": kl_shift,
                "kl_repo_style": kl_repo,
                "kl_repo_style_real_vs_real_floor": kl_repo_floor,
                "ssim": ssim,
                "abs_return_acf": {
                    "real": acf_real.tolist(),
                    "generated": acf_fake.tolist(),
                },
            },
            fh,
            indent=2,
        )

    plot_heatmaps(heatmap_levels(real_raw), heatmap_levels(fake_raw))
    plot_acf(acf_real, acf_fake)
    print("\nSaved out/metrics.json, figs/depth_heatmaps.png, figs/abs_return_acf.png")


if __name__ == "__main__":
    main()
