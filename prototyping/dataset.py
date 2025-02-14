import os
import torch
from torch.utils.data import Dataset
from typing import Literal
from song_pipeline.constants import DATA_DIR


class BatchedDataset(Dataset):
    def __init__(self, set_label: Literal['train', 'valid'],):
        """
        Args:
            set_label (Literal['train', 'valid']): Specifies whether the data belongs to the training or validation set.r.
        """
        if set_label == 'train':
            self.path = os.path.join(DATA_DIR, 'train')
        else:
            self.path = os.path.join(DATA_DIR, 'valid')

    def __len__(self):
        return len(os.listdir(self.path)) // 2  # division by two due to Xs and Ys being stored in one folder

    def __getitem__(self, idx):
        X = torch.load(os.path.join(self.path, f'X_{idx}.pt'))
        Y = torch.load(os.path.join(self.path, f'Y_{idx}.pt'))
        return X.float(), Y.float()

    @staticmethod
    def collate(batch):
        return batch[0], batch[1]
