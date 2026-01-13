import torch
import torchaudio.transforms as T
import random


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

    def __init__(self, augmentations_p: float, augmentations: list[dict]):
        self.transforms = []
        self.p = augmentations_p

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
        augmented_specs = []
        for spec in spectrograms:
            augmented_spec = spec
            for t in self.transforms:
                if random.random() < self.p:
                    augmented_spec = t(augmented_spec, mask_value=-80.0)
            augmented_specs.append(augmented_spec)
        return augmented_specs
