import torch
import torchaudio.transforms as T


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

    def __init__(self, augmentations: list[dict]):
        self.transforms = []

        for aug_cfg in augmentations:
            name = aug_cfg["name"]
            params = aug_cfg["params"]
            AugCls = self._AUG_MAP[name]
            transform = AugCls(**params)
            self.transforms.append(transform)

    def __call__(
        self,
        spectrograms: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        spectrograms (list[torch.Tensor]): spectrograms of shape [1, n_mels, len]
        """
        if not spectrograms:
            return []

        batch = torch.stack(spectrograms, dim=0)  # [batch, 1, n_mels, len]
        for t in self.transforms:
            batch = t(batch)
        return [b for b in batch]
