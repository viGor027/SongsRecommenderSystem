from typing import TypedDict, List
import numpy as np


class MelSpecKwargsType(TypedDict):
    """
    Represents the keyword arguments for generating mel spectrograms.

    Attributes:
        sr (int): The sample rate of the audio fragments.
        n_mels (int): The number of mel bands used for mel spectrogram computation.
    """
    sr: int
    n_mels: int


class ConfigType(TypedDict):
    """
    Represents the configuration settings of the spectrogram pipeline.

    Attributes:
        n_mels (int): The number of mel bands used for mel spectrogram computation.
        n_seconds (int): The duration (in seconds) of each audio fragment.
        spec_type (int): The type of spectrogram being used:
            - 0 for standard spectrogram.
            - 1 for mel spectrogram.
    """
    n_mels: int
    n_seconds: int
    spec_type: int


class SongSpecDataDictType(TypedDict):
    """
    Represents the spectrogram and metadata for a single song.

    Attributes:
        title (str): The title of the song.
        samples (List[np.ndarray]): A list of spectrogram fragments, where each fragment is a numpy array.
        tags (List[str]): A list of tags or labels associated with the song.
    """
    title: str
    samples: List[np.ndarray]
    tags: List[str]
