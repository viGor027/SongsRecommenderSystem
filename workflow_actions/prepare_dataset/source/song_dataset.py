import torch
from torch.utils.data import Dataset
from typing import Literal
from workflow_actions.paths import MODEL_READY_DIR


class SongDataset(Dataset):
    """
    Assumes the following directory layout:
        data/
            model_ready/
                train/
                    X_0.pt, Y_0.pt, ...
                valid/
                    X_0.pt, Y_0.pt, ...

    Notes:
        - each X_i.pt and Y_i.pt file contains one batch.
    """
    def __init__(self, dataset_type: Literal['train', 'valid']):
        """
        Initialize the dataset with paths to batches of features (X) and labels (Y).

        Args:
            dataset_type (str): Either 'train' or 'valid'
        """
        paths = {
            'train': MODEL_READY_DIR / 'train',
            'valid': MODEL_READY_DIR / 'valid'
        }

        self.data_path = paths[dataset_type]
        self.x_files = sorted(self.data_path.glob("X_*.pt"))
        self.y_files = sorted(self.data_path.glob("Y_*.pt"))

        assert len(self.x_files) == len(self.y_files), "Mismatch between X and Y batch files"

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        return torch.load(self.x_files[idx]), torch.load(self.y_files[idx])


if __name__ == "__main__":
    test_dir = MODEL_READY_DIR / "train"
    test_dir.mkdir(parents=True, exist_ok=True)

    dataset = SongDataset("train")
    print(len(dataset))
    print(dataset[0])
    print(type(dataset[0]), len(dataset[0]))
