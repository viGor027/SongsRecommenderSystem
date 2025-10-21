from torch.utils.data import Dataset
from typing import Literal
from workflow_actions.paths import MODEL_READY_DATA_DIR
import torch
import os


class FragmentsDataset(Dataset):
    """
    Implements dataset for **model ready** fragments.
    """

    def __init__(self, dataset_type: Literal["train", "valid"]):
        self.data_path = MODEL_READY_DATA_DIR / dataset_type

        self.n_elements_in_data_dir = len(os.listdir(self.data_path))
        if self.n_elements_in_data_dir % 2 == 1:
            raise RuntimeError(
                f"Number of elements in {self.data_path} is odd. Check data correctness."
            )

    def __len__(self):
        return self.n_elements_in_data_dir // 2

    def __getitem__(self, idx):
        X_path = self.data_path / f"X_{idx}.pt"
        y_path = self.data_path / f"y_{idx}.pt"
        return torch.load(X_path), torch.load(y_path)

    @staticmethod
    def collate_concat(batch):
        xs, ys = zip(*batch)
        xs = torch.cat(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        return xs, ys
