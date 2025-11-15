from torch.utils.data import Dataset
from typing import Literal
from workflow_actions.paths import MODEL_READY_DATA_DIR
import torch


class FragmentsDataset(Dataset):
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
                self.data_path.glob("sample_*.pt"),
                key=lambda p: int(p.stem.split("_", 1)[1]),
            )
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        d = torch.load(path, map_location="cpu")
        return d["X"], d["y"]

    @staticmethod
    def collate_concat(batch):
        xs, ys = zip(*batch)
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        return xs, ys
