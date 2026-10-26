"""Decoder-side calibration of generated features to the real data's lattice.

Three structural tells separate generated books from real ones almost
perfectly (see tells.py), none of which the continuous generator learns:

1. EVENT SPARSITY. Real books are exactly constant between most events (a
   level's volume is unchanged on ~88% of steps). A "hold filter" keeps each
   series at its previous value unless it moves by more than a threshold tau,
   calibrated per series so the share of unchanged steps matches real data.
2. ROUND LOTS. About half of real level volumes are multiples of 100 and ~59%
   multiples of 10; generated volumes are arbitrary integers (0.3% / 9.6%).
   Held volume segments are snapped to a multiple of 100 / 10 with
   probabilities calibrated to the real shares.
3. ONE-TICK GAPS. ~70% of adjacent levels are exactly one tick apart; the
   continuous generator gives ~35%. Deeper-level gaps below a threshold are
   collapsed to one tick, the threshold calibrated to the real share.

Limits: these match marginal frequencies, not which levels change together or
why. This is post-processing for sampling, not a change to the model.
"""

import numpy as np

from dataset import MID_FEATURE

N_LEVELS = 10
VOL_IDX = [4 * l + k for l in range(N_LEVELS) for k in (1, 3)]  # ask, bid log volume
GAP_IDX = [4 * l + k for l in range(N_LEVELS) for k in (0, 2)]  # ask, bid gap
GAP_L1 = np.array([True, True] + [False] * (2 * N_LEVELS - 2))  # level-1 gaps: half ticks
N_VOL, N_GAP = len(VOL_IDX), len(GAP_IDX)


def _round_vol(x):
    return np.rint(np.expm1(np.maximum(x, 0.0)))


def _round_gap(x):
    return np.where(GAP_L1, np.rint(2.0 * x) / 2.0, np.rint(x))


def _round_mid(m):
    return np.rint(np.sinh(m) * 2.0) / 2.0


def unchanged_share(features):
    """(N, T, 42) -> (41,): share of steps with no change after rounding for
    the 20 volumes, 20 gaps and the midprice change (share of exact zeros)."""
    f = np.asarray(features, dtype=np.float64)
    vol = _round_vol(f[..., VOL_IDX])
    gap = _round_gap(f[..., GAP_IDX])
    mid = _round_mid(f[..., MID_FEATURE])
    s_vol = (vol[:, 1:] == vol[:, :-1]).mean(axis=(0, 1))
    s_gap = (gap[:, 1:] == gap[:, :-1]).mean(axis=(0, 1))
    s_mid = (mid[:, 1:] == 0.0).mean()
    return np.concatenate([s_vol, s_gap, [s_mid]])


def lattice_stats(features):
    """Share of non-empty volumes that are multiples of 100 and of 10, and
    the share of deeper-level (2..10) gaps that are exactly one tick."""
    f = np.asarray(features, dtype=np.float64)
    vol = _round_vol(f[..., VOL_IDX])
    nz = vol > 0
    n = max(nz.sum(), 1)
    gap = np.maximum(1.0, np.rint(f[..., GAP_IDX][..., 2:]))
    ne = nz[..., 2:]
    return {
        "mult100": float(((vol % 100 == 0) & nz).sum() / n),
        "mult10": float(((vol % 10 == 0) & nz).sum() / n),
        "gap1": float(((gap == 1.0) & ne).sum() / max(ne.sum(), 1)),
    }


def _hold(x, tau):
    """x (N, T, K), tau (K,): keep the previous output unless |x - prev| > tau."""
    out = x.copy()
    for t in range(1, x.shape[1]):
        keep = np.abs(x[:, t] - out[:, t - 1]) <= tau
        out[:, t] = np.where(keep, out[:, t - 1], x[:, t])
    return out


def _search(share_fn, target, hi, iters=22):
    """Smallest value in [0, hi] whose share reaches the target (monotone)."""
    lo = np.zeros_like(target)
    hi = np.full_like(target, hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = share_fn(mid) < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return hi


def _snap(vol, p100, p10, u):
    """Snap each held volume segment to a multiple of 100 (prob p100) or of 10
    (prob p10). u: uniform draws (same shape); a segment shares one draw."""
    u = u.copy()
    for t in range(1, vol.shape[1]):
        same = vol[:, t] == vol[:, t - 1]
        u[:, t] = np.where(same, u[:, t - 1], u[:, t])
    out = vol.copy()
    s100 = (u < p100) & (vol >= 100)
    out = np.where(s100, np.maximum(100.0, np.rint(vol / 100.0) * 100.0), out)
    s10 = (u >= p100) & (u < p100 + p10) & (vol >= 10)
    return np.where(s10, np.rint(vol / 10.0) * 10.0, out)


def _share(vol, mod):
    nz = vol > 0
    return float(((vol % mod == 0) & nz).sum() / max(nz.sum(), 1))


def calibrate(generated, real, seed=0):
    """Fit all post-processing parameters on generated samples so their
    statistics match `real` (both (N, T, 42) features in original units)."""
    g = np.asarray(generated, dtype=np.float64)
    target = unchanged_share(real)
    lat = lattice_stats(real)
    t_vol, t_gap, t_mid = target[:N_VOL], target[N_VOL: N_VOL + N_GAP], target[-1]
    vol, gap, mid = g[..., VOL_IDX], g[..., GAP_IDX], g[..., MID_FEATURE]

    def vol_share(tau):
        v = _round_vol(_hold(vol, tau))
        return (v[:, 1:] == v[:, :-1]).mean(axis=(0, 1))

    def gap_share(tau):
        h = _round_gap(_hold(gap, tau))
        return (h[:, 1:] == h[:, :-1]).mean(axis=(0, 1))

    def mid_share(tau):
        m = np.where(np.abs(mid) < tau[0], 0.0, mid)
        return np.array([(_round_mid(m)[:, 1:] == 0.0).mean()])

    tau_vol = _search(vol_share, t_vol, hi=6.0)
    tau_gap = _search(gap_share, t_gap, hi=40.0)
    tau_mid = _search(mid_share, np.array([t_mid]), hi=4.0)

    # round lots on the held volumes
    held_vol = _round_vol(_hold(vol, tau_vol))
    u = np.random.default_rng(seed).random(held_vol.shape)
    p100 = float(_search(
        lambda p: np.array([_share(_snap(held_vol, p[0], 0.0, u), 100)]),
        np.array([lat["mult100"]]), hi=1.0)[0])
    p10 = float(_search(
        lambda p: np.array([_share(_snap(held_vol, p100, p[0], u), 10)]),
        np.array([lat["mult10"]]), hi=max(1.0 - p100, 1e-6))[0])
    snapped = _snap(held_vol, p100, p10, u)

    # one-tick gaps at the deeper levels
    held_gap = _hold(gap, tau_gap)[..., 2:]
    ne = snapped[..., 2:] > 0

    def gap1_share(t):
        c = np.maximum(1.0, np.rint(np.where(held_gap < t[0], 1.0, held_gap)))
        return np.array([((c == 1.0) & ne).sum() / max(ne.sum(), 1)])

    t_gap1 = float(_search(gap1_share, np.array([lat["gap1"]]), hi=8.0)[0])

    return {
        "tau": np.concatenate([tau_vol, tau_gap, tau_mid]),
        "p100": p100,
        "p10": p10,
        "gap1_threshold": t_gap1,
    }


def apply(features, params, seed=None):
    """Hold filter, round-lot snapping and one-tick gap collapse."""
    tau = params["tau"]
    f = np.array(features, dtype=np.float64, copy=True)
    f[..., VOL_IDX] = _hold(f[..., VOL_IDX], tau[:N_VOL])
    f[..., GAP_IDX] = _hold(f[..., GAP_IDX], tau[N_VOL: N_VOL + N_GAP])
    m = f[..., MID_FEATURE]
    f[..., MID_FEATURE] = np.where(np.abs(m) < tau[-1], 0.0, m)

    vol = _round_vol(f[..., VOL_IDX])
    u = np.random.default_rng(seed).random(vol.shape)
    vol = _snap(vol, params["p100"], params["p10"], u)
    f[..., VOL_IDX] = np.log1p(vol)

    gaps = f[..., GAP_IDX]
    deeper = gaps[..., 2:]
    collapse = (deeper < params["gap1_threshold"]) & (vol[..., 2:] > 0)
    gaps[..., 2:] = np.where(collapse, 1.0, deeper)
    f[..., GAP_IDX] = gaps
    return f
