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
        df = read_csv(path)
        return torch.tensor(df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages, orderbook = self.samples[idx]

        return {
            "message": messages,
            "orderbook": orderbook,
        }
