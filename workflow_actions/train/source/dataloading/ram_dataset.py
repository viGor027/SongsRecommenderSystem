from torch.utils.data import Dataset
from typing import Literal
from workflow_actions.paths import MODEL_READY_DATA_DIR
import torch


class RamDataset(Dataset):
    """
    Implements dataset for **model ready** fragments.
    """

    def __init__(self, dataset_type: Literal["train", "valid"]):
        self.data_path = MODEL_READY_DATA_DIR / dataset_type

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_path}")

        self.files = [
            str(path)
            for path in sorted(
                self.data_path.glob("shard_*.pt"),
                key=lambda p: int(p.stem.split("_", 1)[1]),
            )
        ]
        self.X = []
        self.y = []
        for shard_path in self.files:
            shard = torch.load(shard_path)
            self.X.append(shard["X"])
            self.y.append(shard["y"])
        self.X = torch.cat(self.X, dim=0)
        self.y = torch.cat(self.y, dim=0)

    def __len__(self):
        return self.X.size(dim=0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def collate_concat(batch):
        xs, ys = zip(*batch)
        xs = torch.stack(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        return xs, ys
