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


if __name__ == "__main__":
    from workflow_actions.paths import DOWNLOAD_DIR
    import soundfile as sf
    import librosa

    song_path = DOWNLOAD_DIR / "3rd_Prototype,Emdi-House.mp3"

    sr = 22050
    song, sample_rate = librosa.load(
        song_path,
        sr=sr,
    )

    aug_cfg = [
        {
            "name": "PitchShift",
            "params": {"min_semitones": -4, "max_semitones": 4, "p": 1.0},
        },
        {
            "name": "AddGaussianNoise",
            "params": {"min_amplitude": 0.003, "max_amplitude": 0.02, "p": 1.0},
        },
        {
            "name": "Mp3Compression",
            "params": {"min_bitrate": 8, "max_bitrate": 64, "p": 1.0},
        },
    ]

    augmenter = RawAugment(aug_cfg)
    base_name = song_path.stem
    for cfg, transform in zip(aug_cfg, augmenter.transforms):
        name = cfg["name"]
        print(f"Applying augmentation: {name}")

        frag = song.copy()
        if np.random.rand() <= getattr(transform, "p", 1.0):
            augmented = transform(samples=frag, sample_rate=sr)
        else:
            augmented = frag

        out_path = f"{base_name}_{name}.wav"
        sf.write(out_path, augmented, sr)

        print(f"Saved: {out_path}")
