from pathlib import Path

import torch
from pandas import read_csv
from torch.utils.data import Dataset


class LOBSTERDataset(Dataset):
    """
    Sequence-based LOBSTER dataset.

    Each item is a temporal window:
        message:   [seq_len, 6]
        orderbook: [seq_len, 4 * levels]

    Time filtering is performed before windows are created, so no
    sequence can cross a train/validation/test boundary.
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

            messages = self._load_csv(message_file)
            orderbook = self._load_csv(orderbook_file)

            if len(messages) != len(orderbook):
                raise ValueError(
                    f"Row mismatch:\n"
                    f"  {message_file}: {len(messages)}\n"
                    f"  {orderbook_file}: {len(orderbook)}"
                )

            timestamps = messages[:, 0]

            mask = torch.ones(len(messages), dtype=torch.bool)

            if start_time is not None:
                mask &= timestamps >= start_time

            if end_time is not None:
                mask &= timestamps < end_time

            messages = messages[mask]
            orderbook = orderbook[mask]

            # Not enough data to create even one sequence
            if len(messages) < seq_len:
                print(
                    f"Skipping {message_file.parent.name}: "
                    f"only {len(messages)} rows after filtering"
                )
                continue

            # Create sequence start indices
            starts = range(0, len(messages) - seq_len + 1)

            for start in starts:
                end = start + seq_len

                self.samples.append(
                    (
                        messages[start:end],
                        orderbook[start:end],
                    )
                )

            print(
                f"Loaded {message_file.parent.name}: "
                f"{len(messages)} rows -> "
                f"{len(messages) - seq_len + 1} sequences"
            )

    @staticmethod
    def _load_csv(path):
        df = read_csv(path, header=None)
        return torch.tensor(df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages, orderbook = self.samples[idx]

        return {
            "message": messages,
            "orderbook": orderbook,
        }


class LOBFeatureTransform:
    """Convert raw LOBSTER Level-10 order books into model features.

    Raw columns per level:
        ask_price, ask_volume, bid_price, bid_volume

    Output columns per level:
        ask_price_offset_ticks, log_ask_volume,
        bid_price_offset_ticks, log_bid_volume

        Args:
            tick_size: time length of each trainable sequence
    """

    def __init__(self, tick_size=100.0):
        # LOBSTER prices are scaled by 1e4.
        # For AMZN's $0.01 tick, tick_size = 0.01 * 1e4 = 100.
        self.tick_size = tick_size

        self.mean = None
        self.std = None

    def transform_raw(self, orderbook):
        """Transform the raw orderbook data into Tensors for training

        Args:
            orderbook (): Tensor [B, T, 40] or [T, 40]

        Returns:
            Tensor with the same leading dimensions, ending in 40 features.

        Raises:
            ValueError: when orderbook has invalid rows
        """
        x = orderbook.reshape(*orderbook.shape[:-1], 10, 4)

        ask_price = x[..., 0]
        ask_volume = x[..., 1]
        bid_price = x[..., 2]
        bid_volume = x[..., 3]

        # Empty levels have zero volume. This also prevents dummy
        # sentinel prices from entering the price calculations.
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

        # Locate the first occupied ask and bid level.
        ask_indices = torch.arange(10, device=x.device).expand_as(ask_valid)
        bid_indices = torch.arange(10, device=x.device).expand_as(bid_valid)

        first_ask_idx = ask_indices.masked_fill(~ask_valid, 10).amin(dim=-1)
        first_bid_idx = bid_indices.masked_fill(~bid_valid, 10).amin(dim=-1)

        has_quotes = (first_ask_idx < 10) & (first_bid_idx < 10)

        best_ask = ask_price.gather(
            -1, first_ask_idx.clamp(max=9).unsqueeze(-1)
        ).squeeze(-1)

        best_bid = bid_price.gather(
            -1, first_bid_idx.clamp(max=9).unsqueeze(-1)
        ).squeeze(-1)

        mid = 0.5 * (best_ask + best_bid)

        if not has_quotes.all():
            raise ValueError(
                "Found an order-book row without both a valid "
                "ask and bid. Filter or inspect these rows first."
            )

        # Relative price distances, in ticks.
        ask_offset = (ask_price - mid.unsqueeze(-1)) / self.tick_size
        bid_offset = (mid.unsqueeze(-1) - bid_price) / self.tick_size

        # Replace empty/sentinel levels with zero price offset.
        ask_offset = torch.where(
            ask_valid, ask_offset, torch.zeros_like(ask_offset)
        )
        bid_offset = torch.where(
            bid_valid, bid_offset, torch.zeros_like(bid_offset)
        )

        # Log-transform volumes. Empty levels become log(1 + 0) = 0.
        log_ask_volume = torch.log1p(ask_volume.clamp(min=0))
        log_bid_volume = torch.log1p(bid_volume.clamp(min=0))

        features = torch.stack(
            [
                ask_offset,
                log_ask_volume,
                bid_offset,
                log_bid_volume,
            ],
            dim=-1,
        )

        return features.flatten(start_dim=-2)

    def fit(self, feature_batches):
        """
        Fit standardisation statistics using training features

        feature_batches yields unstandardised tensors with the
        feature dimension last.
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
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() on training data first.")

        mean = self.mean.to(features.device)
        std = self.std.to(features.device)

        return (features - mean) / std

    def inverse_normalise(self, features):
        if self.mean is None or self.std is None:
            raise RuntimeError("Call fit() on training data first.")

        mean = self.mean.to(features.device)
        std = self.std.to(features.device)

        return features * std + mean
