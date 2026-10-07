from pathlib import Path

from torch import tensor, float32
from pandas import read_csv
from torch.utils.data import Dataset


class LOBSTERDataset(Dataset):
    """Data loader for LOBSTER dataset"""

    def __init__(self, root="LOBSTER_DATA", seq_len=100):
        self.root = Path(root)
        self.seq_len = seq_len
        self.samples = []

        message_files = sorted(self.root.rglob("*message*.csv"))

        for message_file in message_files:

            orderbook_name = message_file.name.replace("message", "orderbook")
            orderbook_file = message_file.with_name(orderbook_name)

            if not orderbook_file.exists():
                continue

            messages = self._load_csv(message_file)
            orderbook = self._load_csv(orderbook_file)

            if len(messages) != len(orderbook):
                raise ValueError(
                    f"Row mismatch: {message_file} "
                    f"({len(messages)}) vs {orderbook_file}"
                )

            self.samples.append(
                {
                    "messages": messages,
                    "orderbook": orderbook,
                    "symbol": message_file.parent.name,
                }
            )

        # Generate valid sequence starting positions
        self.indices = []

        for day, sample in enumerate(self.samples):
            n = len(sample["messages"])

            for start in range(0, n - seq_len + 1, seq_len):
                self.indices.append((day, start))

    @staticmethod
    def _load_csv(path):
        df = read_csv(path)
        return tensor(df.values, dtype=float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        day, start = self.indices[idx]

        sample = self.samples[day]

        end = start + self.seq_len

        return {
            "message": sample["messages"][start:end],
            "orderbook": sample["orderbook"][start:end],
        }
