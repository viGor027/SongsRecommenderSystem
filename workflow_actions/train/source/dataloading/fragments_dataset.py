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

        if self.n_elements_in_data_dir % 2 == 1:
            raise RuntimeError(
                f"Number of elements in {self.data_path} is odd. Check data correctness."
            )

        x_files = {f.stem.split("_")[1] for f in self.data_path.glob("X_*.pt")}
        y_files = {f.stem.split("_")[1] for f in self.data_path.glob("y_*.pt")}

        missing_x = y_files - x_files
        missing_y = x_files - y_files

        if missing_x or missing_y:
            msg = f"Data mismatch in {self.data_path}."
            if missing_x:
                msg += f" Missing X files for indices: {sorted(missing_x)}."
            if missing_y:
                msg += f" Missing y files for indices: {sorted(missing_y)}."
            raise RuntimeError(msg)

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
