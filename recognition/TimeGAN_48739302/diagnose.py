"""Diagnostics for a trained TimeGAN: is the generator (a) flickering,
(b) ignoring its noise, (c) stuck in a recurrent limit cycle?

Run after train.py:  python diagnose.py
Compares generated vs real test windows on:
    1. lag-1 / lag-2 autocorrelation per feature group  (flicker => lag-1 < 0)
    2. mean |step-to-step change|, generated / real       (roughness, want ~1)
    3. roughness by time block                            (does flicker start at a fixed t?)
    4. diversity across samples                           (template/mode collapse)
    5. noise sensitivity                                  (is z actually used?)
    6. latent smoothness: real h vs generated e_hat, h_hat, x_hat
"""

import sys

import numpy as np
import torch

from dataset import DT_COL, N_FEATURES, LOBSTERDataset
from predict import (
    BURN_IN,
    DEVICE,
    SEED,
    TICK_SIZE,
    decode_features,
    SEQ_LEN,
    TEST_END,
    TEST_START,
    generate_sequences,
    load_model,
)

N = 500
GROUPS = {
    "log volumes": [i for i in range(40) if i % 4 in (1, 3)],
    "level gaps": [i for i in range(40) if i % 4 in (0, 2)],
    "midprice change (asinh)": [40],
    "log inter-event dt": [41],
}


def lag_autocorr(x, lag):
    """x (N, T, D) -> (D,) mean within-window autocorrelation at `lag`."""
    xc = x - x.mean(axis=1, keepdims=True)
    den = (xc ** 2).sum(axis=1)
    num = (xc[:, :-lag] * xc[:, lag:]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = np.where(den > 1e-12, num / den, np.nan)
    return np.nanmean(ac, axis=0)


def roughness(x):
    """(D,) mean |x_t - x_{t-1}|."""
    return np.abs(np.diff(x, axis=1)).mean(axis=(0, 1))


def pairwise_corr(series):
    """series (N, T): mean correlation between different samples at the
    same time index (0 = independent, 1 = identical template)."""
    s = series[:50]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(s)  # constant series give NaN and are skipped
    return float(np.nanmean(c[np.triu_indices(len(s), 1)]))


def load_real(transform, start=TEST_START, end=TEST_END):
    ds = LOBSTERDataset("LOBSTER_DATA", SEQ_LEN, start, end)
    idx = np.linspace(0, len(ds) - 1, N).astype(int)
    ob = torch.stack([ds[int(i)]["orderbook"] for i in idx])
    ms = torch.stack([ds[int(i)]["message"] for i in idx])
    raw = transform.transform_raw(ob, ms)
    start_mid = (0.5 * (ob[:, 0, 0].double() + ob[:, 0, 2].double()) / TICK_SIZE).numpy()
    return raw.numpy(), transform.normalise(raw), start_mid


def spread_series(f):
    return f[..., 0] + f[..., 2]  # ask_gap_1 + bid_gap_1, ticks


@torch.no_grad()
def main():
    torch.manual_seed(SEED)
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    model, transform = load_model(ckpt) if ckpt else load_model()
    real, real_norm, start_mid = load_real(transform)
    gen = generate_sequences(model, transform, N, SEQ_LEN)

    # Compare reconstruction before attributing lost event dynamics to G/S.
    from tells import window_stats
    with torch.no_grad():
        reconstruction = transform.inverse_normalise(
            model.reconstruct(real_norm.to(DEVICE)).cpu()
        ).numpy()
    event_keys = [
        "bid3: share unchanged", "bid3: mean |dlogvol| when changed",
        "bid3: std log vol", "imbalance L1: std", "log dt: std",
    ]
    print("Event diagnostics: per-window medians (real | reconstructed | generated)")
    event_stats = [window_stats(decode_features(f, start_mid))
                   for f in (real, reconstruction, gen)]
    for key in event_keys:
        values = [np.nanmedian(stats[key]) for stats in event_stats]
        print(f"   {key:<42s}" + " ".join(f"{v:10.3f}" for v in values))

    print("1) autocorrelation (real | generated)")
    print(f"   {'group':<20s} {'lag1 real':>10s} {'lag1 gen':>10s} "
          f"{'lag2 real':>10s} {'lag2 gen':>10s}")
    for name, idx in GROUPS.items():
        r1, g1 = lag_autocorr(real[..., idx], 1).mean(), lag_autocorr(gen[..., idx], 1).mean()
        r2, g2 = lag_autocorr(real[..., idx], 2).mean(), lag_autocorr(gen[..., idx], 2).mean()
        print(f"   {name:<20s} {r1:10.3f} {g1:10.3f} {r2:10.3f} {g2:10.3f}")
    print("   generated lag1 far below real = step-to-step flicker")

    print("\n2) roughness  mean|dx| generated / real  (want ~1)")
    rr, rg = roughness(real), roughness(gen)
    for name, idx in GROUPS.items():
        print(f"   {name:<20s} {np.mean(rg[idx] / np.maximum(rr[idx], 1e-9)):8.2f}")

    print("\n3) spread roughness by time block (ticks)")
    blocks = range(0, SEQ_LEN, 10)
    d_gen = np.abs(np.diff(spread_series(gen), axis=1))
    d_real = np.abs(np.diff(spread_series(real), axis=1))
    print("   t        " + " ".join(f"{b:5d}" for b in blocks))
    print("   real     " + " ".join(f"{d_real[:, b:b+10].mean():5.2f}" for b in blocks))
    print("   generated" + " ".join(f"{d_gen[:, b:b+10].mean():5.2f}" for b in blocks))

    print("\n4) diversity across samples")
    for name, fn in (("spread", spread_series), ("L1 ask log-vol", lambda f: f[..., 1])):
        print(f"   {name:<16s} pairwise corr at same t: "
              f"real {pairwise_corr(fn(real)):5.2f}  generated {pairwise_corr(fn(gen)):5.2f}")
    print("   (generated >> real means every sample follows the same time template)")

    print("\n5) noise sensitivity (spread series, per-sample correlation)")
    model.eval()
    z1 = model.sample_noise(N, SEQ_LEN + BURN_IN, DEVICE)
    z2 = model.sample_noise(N, SEQ_LEN + BURN_IN, DEVICE)
    zc = torch.full_like(z1, 0.5)

    def decode(z):
        gen_net, sup_net, rec_net = model.nets()
        e = gen_net(z)
        h = sup_net(e)
        x = rec_net(h)
        e, h, x = e[:, BURN_IN:], h[:, BURN_IN:], x[:, BURN_IN:]  # drop warm-up
        return e, h, x, spread_series(transform.inverse_normalise(x.cpu()).numpy())

    e1, h1, x1, s1 = decode(z1)
    _, _, _, s2 = decode(z2)
    _, _, _, sc = decode(zc)

    def mean_corr(a, b):
        c = [np.corrcoef(a[i], b[i])[0, 1] for i in range(len(a))]
        return float(np.nanmean(c))

    print(f"   z1 vs z2 (different noise): {mean_corr(s1, s2):5.2f}")
    print(f"   z1 vs constant noise:       {mean_corr(s1, sc):5.2f}")
    print("   ~1 => the noise is (almost) ignored; a recurrent limit cycle or")
    print("   a fixed template produces the output")

    print("\n7) dispersion, cross-feature structure, drift")
    sp_r, sp_g = spread_series(real), spread_series(gen)
    print(f"   spread std (ticks):  real {sp_r.std():6.2f}  generated {sp_g.std():6.2f}")
    for name, i in (("ask_gap_1", 0), ("bid_gap_1", 2)):
        print(f"   {name} std:        real {real[..., i].std():6.2f}  generated {gen[..., i].std():6.2f}")
    cr = np.corrcoef(real[..., 0].ravel(), real[..., 2].ravel())[0, 1]
    cg = np.corrcoef(gen[..., 0].ravel(), gen[..., 2].ravel())[0, 1]
    print(f"   corr(ask_gap_1, bid_gap_1): real {cr:5.2f}  generated {cg:5.2f}")
    corr_r = np.corrcoef(real.reshape(-1, N_FEATURES), rowvar=False)
    corr_g = np.corrcoef(gen.reshape(-1, N_FEATURES), rowvar=False)
    gap = np.nanmean(np.abs(corr_r - corr_g))
    print(f"   mean |corr_real - corr_gen| over all feature pairs: {gap:.3f}")

    def template_share(s):
        """share of variance explained by the across-sample mean path m(t)"""
        return float(s.mean(axis=0).var() / s.var())
    print(f"   time-template share of variance (spread): real "
          f"{template_share(sp_r):.2f}  generated {template_share(sp_g):.2f}")
    print(f"   time-template share (L1 ask log-vol):      real "
          f"{template_share(real[..., 1]):.2f}  generated {template_share(gen[..., 1]):.2f}")
    mr = np.sinh(real[..., 40]).mean()
    mg = np.sinh(gen[..., 40]).mean()
    print(f"   mean midprice change per step (ticks): real {mr:+.3f}  generated {mg:+.3f}")

    print("\n11) event sparsity: does the book stay exactly constant between events?")
    print("    (real and generated features decoded with the same function)")
    dec_r = decode_features(real, start_mid)
    dec_g = decode_features(gen, start_mid)
    real_tr, _, start_tr = load_real(transform, 34_200, 46_800)
    dec_tr = decode_features(real_tr, start_tr)  # training period (09:30-13:00)

    def sparsity(dec):
        vol = np.concatenate([dec["ask_vol"], dec["bid_vol"]], axis=-1)  # (N,T,20)
        price = np.concatenate([dec["ask_price"], dec["bid_price"]], axis=-1)
        vchg = vol[:, 1:] != vol[:, :-1]
        pchg = price[:, 1:] != price[:, :-1]
        return {
            "vol unchanged, all levels": 1 - vchg.mean(),
            "vol unchanged, level 1": 1 - vchg[..., [0, 10]].mean(),
            "vol unchanged, level 10": 1 - vchg[..., [9, 19]].mean(),
            "price unchanged, level 1": 1 - pchg[..., [0, 10]].mean(),
            "price unchanged, level 5": 1 - pchg[..., [4, 14]].mean(),
            "levels with volume change / step (median)": float(np.median(vchg.sum(-1))),
            "levels with volume change / step (95th pct)": float(np.percentile(vchg.sum(-1), 95)),
        }

    sr, sg, st = sparsity(dec_r), sparsity(dec_g), sparsity(dec_tr)
    print(f"   {'':<46s} {'real test':>9s} {'real train':>10s} {'generated':>10s}")
    for k in sr:
        print(f"   {k:<46s} {sr[k]:9.3f} {st[k]:10.3f} {sg[k]:10.3f}")
    print("   generated << real in the 'unchanged' rows means every level jitters")
    print("   every step, while real books change only a few levels per event")

    print("\n6) latent lag-1 autocorrelation (mean over hidden units)")
    h_real = model.embedder(real_norm.to(DEVICE))
    for name, t in (("real h", h_real), ("e_hat (generator)", e1),
                    ("h_hat (supervisor)", h1), ("x_hat (recoverer)", x1)):
        print(f"   {name:<20s} {np.nanmean(lag_autocorr(t.cpu().numpy(), 1)):6.3f}")
    print("   the first stage where this collapses vs 'real h' is where the")
    print("   flicker is introduced")

    print("\n8) latent dispersion (is the generator under-dispersed before the recoverer?)")

    def unit_std(t):
        return t.cpu().numpy().reshape(-1, t.shape[-1]).std(axis=0)

    def saturated(t):
        a = t.cpu().numpy()
        return float(((a > 0.95) | (a < 0.05)).mean())

    s_real = unit_std(h_real)
    for name, t in (("e_hat (generator)", e1), ("h_hat (supervisor)", h1)):
        s = unit_std(t)
        print(f"   {name:<20s} mean unit std {s.mean():.3f} (real h {s_real.mean():.3f}), "
              f"units below half of real: {(s < 0.5 * s_real).mean():.0%}")
    print(f"   fraction of latent values saturated (<0.05 or >0.95): "
          f"real h {saturated(h_real):.1%}  e_hat {saturated(e1):.1%}  h_hat {saturated(h1):.1%}")

    print("\n9) can the decoder reproduce real dispersion? (reconstruction of real windows)")
    x_rec = model.recoverer(h_real)
    rec = transform.inverse_normalise(x_rec.cpu()).numpy()
    print(f"   spread std (ticks): real {spread_series(real).std():.2f}  "
          f"reconstructed {spread_series(rec).std():.2f}  generated {sp_g.std():.2f}")
    print("   reconstructed ~ real but generated << real => the generator, not the")
    print("   decoder, is the bottleneck")

    print("\n10) generation-time noise amplitude (does a bigger drive raise dispersion?)")
    print("    the other repo trains with U[0,1] but generates with mean 0, std 1")
    print("    (U[-1.73, 1.73]), i.e. ~3.5x the training amplitude")
    base = model.sample_noise(N, SEQ_LEN + BURN_IN, DEVICE)
    trials = (
        ("training noise (x1)", base),
        ("x2 around 0.5", 0.5 + 2.0 * (base - 0.5)),
        ("x3.5 around 0.5", 0.5 + 3.5 * (base - 0.5)),
        ("repo-style mean 0, std 1", (base - 0.5) * 12 ** 0.5),
    )
    print(f"   {'noise':<26s} {'spread std':>10s} {'spread q5/50/95':>18s} {'vol lag1':>9s}")
    for label, z in trials:
        _, _, xg, sg = decode(z)
        fg = transform.inverse_normalise(xg.cpu()).numpy()
        q5, q50, q95 = np.percentile(sg, [5, 50, 95])
        v1 = lag_autocorr(fg[..., GROUPS["log volumes"]], 1).mean()
        print(f"   {label:<26s} {sg.std():10.2f} {q5:6.1f}/{q50:4.1f}/{q95:4.1f} {v1:9.3f}")
    print(f"   real: spread std {spread_series(real).std():.2f}, "
          f"q5/50/95 {'/'.join(f'{v:.1f}' for v in np.percentile(spread_series(real), [5, 50, 95]))}")


if __name__ == "__main__":
    main()
