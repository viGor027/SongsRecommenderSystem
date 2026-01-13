import numpy as np
from audiomentations import (
    AddGaussianNoise,
    PitchShift,
    Mp3Compression,
)


class RawAugment:
    """
    Class supporting raw mp3 augmentations.

    To introduce new augmentation add entry in raw_augment.augmentation list in
    prepare_dataset_config.json, this entry needs to contain `name` key which value
    corresponds to key in `_AUG_MAP`and `params` key that is dictionary containing every
    parameter that class from `_AUG_MAP` needs to be initialized with.
    """

    _AUG_MAP = {
        "PitchShift": PitchShift,
        "AddGaussianNoise": AddGaussianNoise,
        "Mp3Compression": Mp3Compression,
    }

    def __init__(self, augmentations: list[dict]):
        """
        log_path (Path or str): path where logs will be saved
        augmentations (list): list of dicts, each with 'name' and a 'params' sub-dictionary
        """
        self.transforms = []

        for aug_cfg in augmentations:
            name = aug_cfg["name"]
            params = aug_cfg["params"]
            AugCls = self._AUG_MAP[name]
            self.transforms.append(AugCls(**params))

    def __call__(
        self,
        raw_fragments: list[np.ndarray],
        sample_rate: int,
    ) -> list[np.ndarray]:
        if not raw_fragments:
            return []

        out: list[np.ndarray] = []
        for frag in raw_fragments:
            for t in self.transforms:
                if np.random.rand() <= getattr(t, "p", 1.0):
                    frag = t(samples=frag, sample_rate=sample_rate)
            out.append(frag)
        return out
