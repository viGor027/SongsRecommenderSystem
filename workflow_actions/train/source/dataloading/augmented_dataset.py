from torch.utils.data import Dataset
from typing import Literal
from workflow_actions.paths import (
    MODEL_READY_DATA_DIR,
    DATASET_PREPROCESSOR_CONFIG_PATH,
)
from workflow_actions.json_handlers import read_json_to_dict
from workflow_actions.dataset_preprocessor.source import FragmentPipeline
import torch


class AugmentedDataset(Dataset):
    """
    Implements dataset for **model ready** fragments.
    """

    _fragment_pipeline_cfg = read_json_to_dict(DATASET_PREPROCESSOR_CONFIG_PATH)[
        "fragment_pipeline"
    ]
    fragment_pipeline = FragmentPipeline(**_fragment_pipeline_cfg)

    def __init__(self, dataset_type: Literal["train", "valid"]):
        self.data_path = MODEL_READY_DATA_DIR / dataset_type

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_path}")

        self.x_files = [
            str(path)
            for path in sorted(
                self.data_path.glob("X_*.pt"),
                key=lambda p: int(p.stem.split("_", 1)[1]),
            )
        ]
        self.y_files = [
            str(path)
            for path in sorted(
                self.data_path.glob("y_*.pt"),
                key=lambda p: int(p.stem.split("_", 1)[1]),
            )
        ]
        if len(self.x_files) != len(self.y_files):
            raise ValueError("Number of X files doesn't match number of y files.")
        self.n_samples = len(self.x_files)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.load(self.x_files[idx]), torch.load(self.y_files[idx])

    @staticmethod
    def collate_concat(batch):
        xs, ys = zip(*batch)
        xs = AugmentedDataset.fragment_pipeline.process_raw_fragments(
            fragments=xs, augment=True
        )
        xs = torch.stack(xs, dim=0)
        ys = torch.stack(ys, dim=0)
        return xs, ys
