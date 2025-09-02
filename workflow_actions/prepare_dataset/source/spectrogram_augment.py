import torch
import numpy as np
import torchaudio.transforms as T
from pathlib import Path


class SpectrogramAugment:
    """
    Spectrogram-level augmentation using torchaudio.transforms.

    To introduce new augmentation add entry in spectrogram_augment.augmentation list in
    prepare_dataset_config.json, this entry needs to contain `name` key which value
    corresponds to key in `_AUG_MAP`and `params` key that is dictionary containing every
    parameter that class from `_AUG_MAP` needs to be initialized with.
    """

    _AUG_MAP = {
        "TimeMasking": T.TimeMasking,
        "FrequencyMasking": T.FrequencyMasking,
    }

    def __init__(self, log_path: Path | str | None, augmentations: list[dict]):
        self.log_path = log_path
        self.transforms = []  # list of (Module, probability)
        self._records = []

        for aug_cfg in augmentations:
            name = aug_cfg["name"]
            params = aug_cfg["params"]
            AugCls = self._AUG_MAP[name]
            transform = AugCls(**params)
            self.transforms.append(transform)

    def __call__(
        self, fragment_id: str, spectrogram: np.ndarray
    ) -> tuple[torch.Tensor, list[str]]:
        """
        fragment_id:  identyfikator
        spectrogram:  2D float32 numpy array, shape (n_mels, time_steps)
        """
        record = {"fragment_id": fragment_id, "applied": []}

        spec = torch.from_numpy(spectrogram.astype(np.float32)).unsqueeze(0)

        for transform in self.transforms:
            spec = transform(spec)
            record["applied"].append(transform.__class__.__name__)

        self._records.append(record)
        return spec, record["applied"]
