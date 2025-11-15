from typing import Literal

import torch
from torch.utils.data import IterableDataset

from workflow_actions.paths import MODEL_READY_DATA_DIR


class ShardsIterableDataset(IterableDataset):
    def __init__(self, dataset_type: Literal["train", "valid"]):
        self.data_path = MODEL_READY_DATA_DIR / dataset_type

        self.shard_files = sorted(
            self.data_path.glob("shard_*.pt"),
            key=lambda p: int(p.stem.split("_", 1)[1]),
        )

    def __iter__(self):
        for shard_path in self.shard_files:
            d = torch.load(shard_path, map_location="cpu")
            X = d["X"]
            y = d["y"]

            n = X.shape[0]
            if n != y.shape[0]:
                raise ValueError(
                    f"X and y have different first dim in {shard_path}: "
                    f"{X.shape[0]} vs {y.shape[0]}"
                )

            perm = torch.randperm(n)

            for i in perm:
                x_i = X[i]
                y_i = y[i]
                yield x_i, y_i
