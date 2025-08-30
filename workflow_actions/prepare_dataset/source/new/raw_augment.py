import numpy as np
from audiomentations import AddGaussianNoise, TimeStretch, PitchShift, Shift, Mp3Compression
from pathlib import Path
from workflow_actions.json_handlers import write_dict_to_json


class RawAugment:
    """
    Class supporting raw mp3 augmentations.

    To introduce new augmentation add entry in raw_augments.augmentation list in
    prepare_dataset_config.json, this entry needs to contain `name` key which value
    corresponds to key in `_AUG_MAP`and `params` key that is dictionary containing every
    parameter that class from `_AUG_MAP` needs to be initialized.
    """

    _AUG_MAP = {
        "AddGaussianNoise": AddGaussianNoise,
        "TimeStretch": TimeStretch,
        "PitchShift": PitchShift,
        "Shift": Shift,
        "Mp3Compression": Mp3Compression,
    }

    def __init__(self, log_path: Path | str | None, augmentations: list[dict]):
        """
        log_path (Path or str): path where logs will be saved
        augmentations (list): list of dicts, each with 'name' and a 'params' sub-dictionary
        """
        self.log_path = log_path
        self.transforms = []
        self._records = []

        for aug_cfg in augmentations:
            name = aug_cfg["name"]
            params = aug_cfg["params"]
            AugCls = self._AUG_MAP[name]
            self.transforms.append(AugCls(**params))

    def __call__(
            self,
            fragment_id: str | None,
            raw_fragment: np.ndarray,
            sample_rate: int
    ) -> tuple[np.ndarray, list[str]]:
        """Augment single song.

        Returns:
            augmented song and list of transformation used during augmentation
        """
        record = {
            "fragment_id": fragment_id,
            "applied": []
        }

        augmented_fragment = raw_fragment
        for transform in self.transforms:
            if np.random.rand() <= transform.p:
                augmented_fragment = transform(samples=augmented_fragment, sample_rate=sample_rate)
                record["applied"].append(transform.__class__.__name__)

        self._records.append(record)
        return augmented_fragment, record["applied"]

    def save_records(self):
        write_dict_to_json(data=self._records, file_path=self.log_path)
