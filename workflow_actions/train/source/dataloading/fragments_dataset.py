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

        data_dir_elems = list(self.data_path.iterdir())
        self.n_elements_in_data_dir = len(data_dir_elems)

    def __len__(self):
        return self.n_elements_in_data_dir

    def __getitem__(self, idx):
        d = torch.load(self.data_path / f"sample_{idx}.pt", map_location="cpu")
        return d["X"], d["y"]

    @staticmethod
    def collate_concat(batch):
        xs, ys = zip(*batch)
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        return xs, ys
