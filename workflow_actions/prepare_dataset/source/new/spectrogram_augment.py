import torch
import numpy as np
import torchaudio.transforms as T
from pathlib import Path


class SpectrogramAugment:
    """
    Spectrogram-level augmentation using torchaudio.transforms.

    Config format (JSON):
    {
      "log_path": "...",
      "augmentations": [
        {
          "name": "TimeMasking",
          "params": {
            "time_mask_param": 80,
            "iid_masks": False,
            "p": 0.5
          }
        },
        {
          "name": "FrequencyMasking",
          "params": {
            "freq_mask_param": 30,
            "iid_masks": False,
          }
        }
      ]
    }
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

    def __call__(self, fragment_id: str, spectrogram: np.ndarray) -> tuple[torch.Tensor, list[str]]:
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
