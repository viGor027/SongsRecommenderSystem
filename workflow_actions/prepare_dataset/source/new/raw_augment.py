import numpy as np
from audiomentations import AddGaussianNoise, TimeStretch, PitchShift, Shift, Mp3Compression
from pathlib import Path
from workflow_actions.json_handlers import write_dict_to_json, read_json_to_dict


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

    def __init__(self, log_path: Path | str, augmentations: list):
        """
        log_path: path where logs will be saved
        augmentations: list of dicts, each with 'name' and a 'params' sub-dictionary
        """
        self.log_path = log_path
        self.transforms = []
        self._records = []

        for aug_cfg in augmentations:
            name = aug_cfg["name"]
            params = aug_cfg["params"]
            AugCls = self._AUG_MAP[name]
            self.transforms.append(AugCls(**params))

    def __call__(self, song_title: str, raw_song: np.ndarray, sample_rate: int) -> tuple[np.ndarray, list[str]]:
        """
        Augment single song.

        song_title: identifier for the track (used in the log)
        raw_song:   1D numpy array of audio samples

        :returns song and list of augmentation applied
        """
        record = {
            "song_title": song_title,
            "applied": []
        }

        augmented_song = raw_song
        for transform in self.transforms:
            if np.random.rand() <= transform.p:
                augmented_song = transform(samples=augmented_song, sample_rate=sample_rate)
                record["applied"].append(transform.__class__.__name__)

        self._records.append(record)
        return augmented_song, record["applied"]

    def save_records(self):
        write_dict_to_json(data=self._records, file_path=self.log_path)


if __name__ == "__main__":
    # Do porównywania brzmienia augmentacji
    import soundfile as sf
    import json
    from loader import load_single_song
    cfg = read_json_to_dict("../../prepare_dataset_config.json")
    song, sample_rate = load_single_song(Path("D:\\Nauka\\Projekty\\SongsRecommenderSystem\\data\\raw\\downloaded_songs\\ALEXYS,Strn_-So_Sweet.mp3"))
    print(type(song), song.shape)
    raw_augment = RawAugment(log_path="raw_augment_logs.json", **cfg["raw_augment"])
    song, list_of_transforms = raw_augment(song_title='ala', raw_song=song, sample_rate=sample_rate)
    print(type(song), song.shape)
    sf.write("AddGaussianNoise.wav", song, sample_rate)
    raw_augment.save_records()