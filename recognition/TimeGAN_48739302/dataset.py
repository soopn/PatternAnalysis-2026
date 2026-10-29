"""Load paired LOBSTER CSV files and construct 42-channel model features.

LOBSTERDataset discovers message files recursively beneath root and pairs each
with the orderbook file obtained by replacing 'message' with 'orderbook'. Files
have no header and must contain matching rows. Each ten-level order-book row
is [ask price, ask size, bid price, bid size] repeated from level 1 to 10.
Prices use LOBSTER units (dollars times 10,000); sizes are shares. Message columns
are time in seconds after midnight, event type, order ID, size, price, direction.
An added seventh column stores the inter-event interval in seconds.

Rows are filtered to [start_time, end_time) before stride-one windows are built.
Windows never cross file or selected-period boundaries and overlap heavily.
The dataset retains tensor views of the loaded arrays in memory; it does not
stream CSV rows or shuffle them. A DataLoader supplies batching/sampling.

LOBFeatureTransform maps raw windows to this per-event layout:
    channels 4*l : 4*l+4, l=0..9:
        ask gap (ticks), log1p ask size, bid gap (ticks), log1p bid size.
        The first gap on each side is its distance from the midprice; deeper
        gaps are increments between consecutive occupied price levels.
    channel 40: asinh(midprice change in ticks), zero at each window's start.
    channel 41: log1p(inter-event seconds * 1,000,000).

Empty/sentinel levels contribute zero gaps; negative sizes are clipped before
log conversion. A row without any valid quote on either side raises ValueError.
The transform does not repair or impose a valid generated quote grid; predict.py
handles generation-time decoding.

Typical pipeline: instantiate time-filtered datasets, transform raw training
batches, fit the scaler only on those training features, then normalise every
split with the same fitted statistics. inverse_normalise() reverses scaling
only; it does not recover raw prices from gaps or undo log/asinh transforms.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from pandas import read_csv
from torch.utils.data import Dataset

N_LEVELS = 10
BOOK_FEATURES = 4 * N_LEVELS
MID_FEATURE = BOOK_FEATURES  # 40
DT_FEATURE = BOOK_FEATURES + 1  # 41
N_FEATURES = BOOK_FEATURES + 2  # 42

# message array columns: time, type, order id, size, price, direction, dt
DT_COL = 6
DT_SCALE = 1e6  # seconds -> microseconds


class LOBSTERDataset(Dataset):
    """
    Sequence-based LOBSTER dataset.

    Args:
        root: Directory containing paired message/orderbook CSV files.
        seq_len: Number of consecutive events in each stride-one window.
        start_time: Inclusive lower bound in seconds after midnight, or None.
        end_time: Exclusive upper bound in seconds after midnight, or None.

    Missing pairs and files shorter than seq_len after filtering are skipped
    with a message. Unequal message/book row counts raise ValueError.

    Each item is a temporal window:
        message:   [seq_len, 7]  (6 LOBSTER columns + dt to the previous event, s)
        orderbook: [seq_len, 4 * levels]

    Time filtering is performed before windows are created, so no
    sequence can cross a train/validation/test boundary.

    The inter-event time dt is computed in float64 on the whole file before
    anything is cast to float32. Raw timestamps (~3e4 s) cannot resolve
    sub-millisecond differences in float32. The first row of each file has
    dt=0; negative intervals are clipped to zero. Filtering retains intervals
    computed before the period boundary, so its first dt may refer to a row
    outside that period. Returned message/book tensors are float32.
    """

    def __init__(
        self,
        root="LOBSTER_DATA",
        seq_len=100,
        start_time=None,
        end_time=None,
    ):
        self.root = Path(root)
        self.seq_len = seq_len
        self.samples = []

        message_files = sorted(self.root.rglob("*message*.csv"))

        for message_file in message_files:
            orderbook_name = message_file.name.replace("message", "orderbook")
            orderbook_file = message_file.with_name(orderbook_name)

            if not orderbook_file.exists():
                print(f"Missing orderbook for {message_file}")
                continue

            messages64 = self._load_csv(message_file, torch.float64)
            orderbook = self._load_csv(orderbook_file)

            if len(messages64) != len(orderbook):
                raise ValueError(
                    f"Row mismatch:\n"
                    f"  {message_file}: {len(messages64)}\n"
                    f"  {orderbook_file}: {len(orderbook)}"
                )

            timestamps = messages64[:, 0]
            dt = torch.diff(timestamps, prepend=timestamps[:1]).clamp(min=0)

            mask = torch.ones(len(messages64), dtype=torch.bool)

            if start_time is not None:
                mask &= timestamps >= start_time

            if end_time is not None:
                mask &= timestamps < end_time

            messages = torch.cat(
                [messages64[:, :6], dt.unsqueeze(1)], dim=1
            ).to(torch.float32)

            messages = messages[mask]
            orderbook = orderbook[mask]

            # Not enough data to create even one sequence
            if len(messages) < seq_len:
                print(
                    f"Skipping {message_file.parent.name}: "
                    f"only {len(messages)} rows after filtering"
                )
                continue

            for start in range(0, len(messages) - seq_len + 1):
                end = start + seq_len
                self.samples.append((messages[start:end], orderbook[start:end]))

            print(
                f"Loaded {message_file.parent.name}: "
                f"{len(messages)} rows -> "
                f"{len(messages) - seq_len + 1} sequences"
            )

    @staticmethod
    def _load_csv(path, dtype=torch.float32):
        df = read_csv(path, header=None)
        return torch.tensor(df.values, dtype=dtype)

    def __len__(self):
        """Return the number of available overlapping event windows."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Return raw message/orderbook tensors for one window.

        These tensors share storage with other overlapping windows; callers
        should avoid in-place modification. Feature conversion is separate.
        """
        messages, orderbook = self.samples[idx]

        return {
            "message": messages,
            "orderbook": orderbook,
        }


class LOBFeatureTransform:
    """Convert raw LOBSTER Level-10 order books + messages into model features.

    Output (N_FEATURES = 42 per timestep), see the layout at the top of file.
    The leading batch dimension is optional. fit() consumes already transformed,
    unstandardised feature batches and stores channel-wise float32 mean/std.
    Statistics broadcast over event/batch dimensions on the input device.

    Args:
        tick_size: price increment in raw LOBSTER units (prices are scaled
            by 1e4, so a $0.01 tick is 100).
    """

    def __init__(self, tick_size=100.0):
        self.tick_size = tick_size
        self.mean = None
        self.std = None

    def transform_raw(self, orderbook, message):
        """
        Args:
            orderbook: Tensor [B, T, 40] or [T, 40]
            message:   Tensor [B, T, 7]  or [T, 7]  (dt in column DT_COL)

        Returns:
            Tensor [..., 42]

        Raises:
            ValueError: when a row has no valid ask or bid
        """
        x = orderbook.reshape(*orderbook.shape[:-1], N_LEVELS, 4)

        ask_price = x[..., 0]
        ask_volume = x[..., 1]
        bid_price = x[..., 2]
        bid_volume = x[..., 3]

        # Empty levels have zero volume; sentinel prices are excluded too.
        ask_valid = (
            (ask_volume > 0)
            & torch.isfinite(ask_price)
            & (ask_price.abs() < 1e9)
        )
        bid_valid = (
            (bid_volume > 0)
            & torch.isfinite(bid_price)
            & (bid_price.abs() < 1e9)
        )

        # First occupied ask and bid level.
        idx = torch.arange(N_LEVELS, device=x.device)
        first_ask_idx = (
            idx.expand_as(ask_valid)
            .masked_fill(~ask_valid, N_LEVELS)
            .amin(dim=-1)
        )
        first_bid_idx = (
            idx.expand_as(bid_valid)
            .masked_fill(~bid_valid, N_LEVELS)
            .amin(dim=-1)
        )

        has_quotes = (first_ask_idx < N_LEVELS) & (first_bid_idx < N_LEVELS)
        if not has_quotes.all():
            raise ValueError(
                "Found an order-book row without both a valid "
                "ask and bid. Filter or inspect these rows first."
            )

        best_ask = ask_price.gather(
            -1, first_ask_idx.clamp(max=N_LEVELS - 1).unsqueeze(-1)
        ).squeeze(-1)
        best_bid = bid_price.gather(
            -1, first_bid_idx.clamp(max=N_LEVELS - 1).unsqueeze(-1)
        ).squeeze(-1)

        mid = 0.5 * (best_ask + best_bid)

        # Distances from the mid, in ticks. Empty levels -> 0.
        ask_offset = ((ask_price - mid.unsqueeze(-1)) / self.tick_size).clamp(
            min=0
        )
        bid_offset = ((mid.unsqueeze(-1) - bid_price) / self.tick_size).clamp(
            min=0
        )
        ask_offset = torch.where(
            ask_valid, ask_offset, torch.zeros_like(ask_offset)
        )
        bid_offset = torch.where(
            bid_valid, bid_offset, torch.zeros_like(bid_offset)
        )

        # Gaps between consecutive levels. Offsets increase with depth, so a
        # running max carries the last valid offset through empty levels and
        # their gap becomes 0. Gap >= 0 by construction.
        ask_run = torch.cummax(ask_offset, dim=-1).values
        bid_run = torch.cummax(bid_offset, dim=-1).values
        ask_gap = (ask_run - F.pad(ask_run[..., :-1], (1, 0))).clamp(min=0)
        bid_gap = (bid_run - F.pad(bid_run[..., :-1], (1, 0))).clamp(min=0)

        log_ask_volume = torch.log1p(ask_volume.clamp(min=0))
        log_bid_volume = torch.log1p(bid_volume.clamp(min=0))

        book = torch.stack(
            [ask_gap, log_ask_volume, bid_gap, log_bid_volume], dim=-1
        ).flatten(
            start_dim=-2
        )  # [..., 40]

        # Midprice change in ticks. First step of a window has no
        # predecessor inside the window, so it is 0.
        d_mid = torch.asinh(
            torch.diff(mid, dim=-1, prepend=mid[..., :1]) / self.tick_size
        )

        # Inter-event time (precomputed in float64 by the dataset).
        dt = message[..., DT_COL].clamp(min=0)
        log_dt = torch.log1p(dt * DT_SCALE)

        return torch.cat(
            [book, d_mid.unsqueeze(-1), log_dt.unsqueeze(-1)], dim=-1
        )

    def fit(self, feature_batches):
        """Fit channel statistics from an iterable of transformed training batches.

        All leading dimensions count as observations; overlapping windows count
        their repeated events again. Accumulation uses float64 and population
        variance, with standard deviations floored at 1e-6. Empty input raises
        ValueError. Returns self; never fit on validation/test features.
        """
        total = None
        total_sq = None
        count = 0

        for features in feature_batches:
            features = features.detach().to(torch.float64)
            features = features.reshape(-1, features.shape[-1])

            batch_sum = features.sum(dim=0)
            batch_sq = features.square().sum(dim=0)

            if total is None:
                total = batch_sum
                total_sq = batch_sq
            else:
                total += batch_sum
                total_sq += batch_sq

            count += features.shape[0]

        if count == 0:
            raise ValueError("Cannot fit scaler on empty training data.")

        mean = total / count
        variance = (total_sq / count - mean.square()).clamp(min=0)

        self.mean = mean.to(torch.float32)
        self.std = variance.sqrt().clamp(min=1e-6).to(torch.float32)

        return self

    def normalise(self, features):
        """Standardise transformed features using fitted channel statistics.

        Preserves shape/device. Raises RuntimeError if statistics are unset.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() on training data first.")

        mean = self.mean.to(features.device)
        std = self.std.to(features.device)

        return (features - mean) / std

    def inverse_normalise(self, features):
        """Undo standardisation, retaining the 42-channel transformed layout.

        Preserves shape/device. Raises RuntimeError if statistics are unset.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() on training data first.")

        mean = self.mean.to(features.device)
        std = self.std.to(features.device)

        return features * std + mean
