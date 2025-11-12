import torch
from typing import Literal
from torch.utils.data import DataLoader
from workflow_actions.paths import MODEL_READY_TRAIN_DIR
from functools import partial


class OfflineNormalizer:
    def __init__(
        self,
        n_mels: int,
        normalization_type=Literal["per_mel"] | None,
    ):
        """Nothing happens when called with normalization_type == None"""
        self.n_mels = n_mels
        compute_norm_map = {
            "per_mel": (
                partial(self.compute_norm_values_per_mel, n_mels=self.n_mels),
                self._apply_compute_norm_values_per_mel,
            )
        }
        self._normalization_type = normalization_type
        self._norm_package = (
            compute_norm_map[self._normalization_type]
            if self._normalization_type
            else tuple()
        )

    def __call__(self):
        if self._normalization_type:
            self._init_train_loader()
            param_function, apply_function = self._norm_package
            params = param_function()
            apply_function(*params)

    def compute_norm_values_per_mel(self, n_mels: int, eps: float = 1e-6):
        """
        Computes mu_f i std_f per mel based on TRAINING SET.
        returns: (mu_f, std_f) both having shape (F,).
        """
        sum_f = torch.zeros(n_mels, dtype=torch.float64)  # (F,)
        sumsq_f = torch.zeros(n_mels, dtype=torch.float64)  # (F,)
        count = 0

        for X, _ in self._train_loader:  # X: (B, F, T)
            x = X.to(torch.float64)
            sum_f += x.sum(dim=(0, 2))  # (F,)
            sumsq_f += (x * x).sum(dim=(0, 2))  # (F,)
            count += x.shape[0] * x.shape[2]  # B * T

        mu_f = (sum_f / count).float()  # (F,)
        var_f = (sumsq_f / count - (sum_f / count) ** 2).clamp_min(0.0)  # (F,)
        std_f = var_f.sqrt().float() + eps  # (F,)
        return mu_f, std_f

    def _apply_compute_norm_values_per_mel(self, mu_f, std_f):
        # from (F,) to (1, F, 1)
        mu = mu_f.view(1, -1, 1)
        std = std_f.view(1, -1, 1)

        # TODO: Implement paralelly
        for p in sorted(MODEL_READY_TRAIN_DIR.glob("X_*.pt")):
            x = torch.load(p)  # (1, F, T) float32
            x = (x - mu) / std  # broadcast po F i T
            torch.save(x, p)

    def _init_train_loader(self):
        from workflow_actions.train.source.dataloading.fragments_dataset import (
            FragmentsDataset,
        )
        self._train_loader = DataLoader(
            FragmentsDataset("train"),
            batch_size=64,
            shuffle=True,
            collate_fn=FragmentsDataset.collate_concat,
            num_workers=0,
            pin_memory=False,
        )