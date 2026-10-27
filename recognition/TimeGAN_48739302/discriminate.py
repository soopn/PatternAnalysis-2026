"""Post-hoc discriminative test

Trains a fresh GRU classifier to tell real test-period windows from generated
ones and reports held-out accuracy. 0.5 means indistinguishable. This does not
depend on the discriminator used during training, which sat at chance in the
training log and so says little about sample quality.

It then repeats the test on each feature group alone, to show which part of
the data still gives the generator away.

Generated features are continuous while real ones are quantised (integer
volumes, tick-sized gaps, half-tick midprice moves), so a classifier can win
on that alone. The main rows therefore re-extract features from the DECODED
book (rounded to the tick grid, same transform as real data); the first row
shows the raw generated features for reference.

Usage: python discriminate.py [checkpoint.pt]
"""

import sys

import numpy as np
import torch
import torch.nn as nn

from dataset import DT_COL, LOBSTERDataset
from diagnose import GROUPS
from predict import (
    DEVICE,
    SEED,
    SEQ_LEN,
    TICK_SIZE,
    TEST_END,
    TEST_START,
    decode_features,
    generate_sequences,
    load_model,
    real_period,
    to_lobster_rows,
)

N_PER_CLASS = 3000
EPOCHS = 8
BATCH = 128
TRAIN_FRAC = 0.7  # real windows are split by time, with a gap, to avoid leakage


class Clf(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.rnn = nn.GRU(d, h, num_layers=2, batch_first=True)
        self.fc = nn.Linear(h, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.fc(o.mean(dim=1)).squeeze(-1)


def auc(scores, labels):
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_real(transform, n):
    a, b = real_period()
    ds = LOBSTERDataset("LOBSTER_DATA", SEQ_LEN, a, b)
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    ob = torch.stack([ds[int(i)]["orderbook"] for i in idx])
    ms = torch.stack([ds[int(i)]["message"] for i in idx])
    start_mid = (
        0.5 * (ob[:, 0, 0].double() + ob[:, 0, 2].double()) / TICK_SIZE
    ).numpy()
    return transform.normalise(transform.transform_raw(ob, ms)), start_mid


def decoded_features(transform, features, start_mid):
    """Generated features -> decoded book -> the same transform as real data."""
    dec = decode_features(features, start_mid)
    raw = to_lobster_rows(dec)
    ok = ((dec["ask_vol"] > 0).any(-1) & (dec["bid_vol"] > 0).any(-1)).all(
        axis=1
    )
    ob = torch.from_numpy(raw[ok]).float()
    msg = torch.zeros(len(ob), ob.shape[1], 7)
    msg[..., DT_COL] = torch.from_numpy(dec["dt"][ok]).float()
    return transform.normalise(transform.transform_raw(ob, msg))


def fit_and_score(real, fake, channels, rng):
    """real/fake: (N, T, D) normalised tensors. Returns metrics dict."""
    n = len(real)
    cut = int(TRAIN_FRAC * n)
    gap = max(1, int(0.05 * n))
    real_tr, real_te = real[:cut], real[cut + gap :]
    perm = rng.permutation(len(fake))
    fake_tr = fake[perm[: len(real_tr)]]
    fake_te = fake[perm[len(real_tr) : len(real_tr) + len(real_te)]]

    def pack(a, b):
        x = torch.cat([a, b])[..., channels]
        y = torch.cat([torch.zeros(len(a)), torch.ones(len(b))])  # 1 = fake
        return x, y

    xtr, ytr = pack(real_tr, fake_tr)
    xte, yte = pack(real_te, fake_te)

    clf = Clf(len(channels)).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        clf.train()
        order = torch.randperm(len(xtr))
        for i in range(0, len(xtr), BATCH):
            b = order[i : i + BATCH]
            opt.zero_grad()
            loss = loss_fn(clf(xtr[b].to(DEVICE)), ytr[b].to(DEVICE))
            loss.backward()
            opt.step()

    clf.eval()
    with torch.no_grad():
        logits = torch.cat(
            [
                clf(xte[i : i + 512].to(DEVICE)).cpu()
                for i in range(0, len(xte), 512)
            ]
        ).numpy()
    y = yte.numpy()
    pred = (logits > 0).astype(float)
    return {
        "accuracy": float((pred == y).mean()),
        "auc": auc(logits, y),
        "real_recall": float((pred[y == 0] == 0).mean()),
        "fake_recall": float((pred[y == 1] == 1).mean()),
    }


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    model, transform = load_model(ckpt) if ckpt else load_model()

    real, start_mid = load_real(transform, N_PER_CLASS)
    fake_feats = generate_sequences(model, transform, N_PER_CLASS, SEQ_LEN)
    fake_rawnorm = transform.normalise(torch.from_numpy(fake_feats).float())
    fake_dec = decoded_features(
        transform, fake_feats, start_mid[rng.permutation(len(start_mid))]
    )

    all_idx = list(range(real.shape[-1]))
    print(
        f"{'features':<34s} {'accuracy':>8s} {'AUC':>6s} {'real recall':>12s} {'fake recall':>12s}"
    )
    runs = [
        ("all, raw generated (reference)", all_idx, fake_rawnorm),
        ("all, decoded", all_idx, fake_dec),
    ]
    runs += [
        (f"{name}, decoded", idx, fake_dec) for name, idx in GROUPS.items()
    ]
    for name, idx, fake in runs:
        m = fit_and_score(real, fake, idx, rng)
        print(
            f"{name:<34s} {m['accuracy']:8.3f} {m['auc']:6.3f} "
            f"{m['real_recall']:12.3f} {m['fake_recall']:12.3f}"
        )
    print(
        "\n0.5 accuracy / 0.5 AUC = indistinguishable. Real windows are split by"
    )
    print(
        "time (train early, test late); a low real recall can also reflect drift"
    )
    print("within the test period rather than generator error.")


if __name__ == "__main__":
    main()
