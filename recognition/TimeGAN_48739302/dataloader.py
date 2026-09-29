import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LOBSTERDataset(Dataset):

    def __init__(self, root="LOBSTER_DATA"):
        self.root = Path(root)

        self.samples = []

        # Find all message CSVs recursively
        message_files = sorted(self.root.rglob("*message*.csv"))

        for message_file in message_files:

            # Find the corresponding orderbook file
            orderbook_name = message_file.name.replace("message", "orderbook")

            orderbook_file = message_file.with_name(orderbook_name)

            if not orderbook_file.exists():
                print(f"Missing orderbook for {message_file}")
                continue

            messages = self._load_csv(message_file)
            orderbook = self._load_csv(orderbook_file)

            if len(messages) != len(orderbook):
                raise ValueError(
                    f"Row mismatch: {message_file} "
                    f"({len(messages)}) vs {orderbook_file} "
                    f"({len(orderbook)})"
                )

            # Store tensors and their source
            self.samples.append(
                {
                    "messages": messages,
                    "orderbook": orderbook,
                    "symbol": message_file.parent.name,
                }
            )

            print(
                f"Loaded {message_file.parent.name}: " f"{len(messages)} rows"
            )

    @staticmethod
    def _load_csv(path):
        rows = []

        with open(path, "r", newline="") as f:
            reader = csv.reader(f)

            for row in reader:
                rows.append([float(x) for x in row])

        return torch.tensor(rows, dtype=torch.float32)

    def __len__(self):
        return sum(len(sample["messages"]) for sample in self.samples)

    def __getitem__(self, idx):

        for sample in self.samples:
            n = len(sample["messages"])

            if idx < n:
                return {
                    "message": sample["messages"][idx],
                    "orderbook": sample["orderbook"][idx],
                }

            idx -= n

        raise IndexError("Index out of range")
