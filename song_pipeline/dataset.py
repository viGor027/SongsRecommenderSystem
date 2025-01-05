import torch
from torch.utils.data import Dataset
import numpy


class SongDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Initialize the dataset with features (X) and labels (Y).

        Parameters:
        - X (torch.Tensor): Song spectrograms.
        - Y (torch.Tensor): Multi-hot encoded tags.
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
