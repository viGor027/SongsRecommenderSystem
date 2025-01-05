from typing import TypedDict, Literal
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
        step (int | float): Defines the interval (in seconds) at which consecutive fragments start within the audio.
        spec_type (Literal['mel', 'std']): The type of spectrogram being used.
                - `'mel'`: Mel spectrogram.
                - `'std'`: Standard spectrogram.
    """
    n_mels: int
    n_seconds: int
    step: int | float
    spec_type: Literal['mel', 'std']


class SongSpecDataDictType(TypedDict):
    """
    Represents the spectrogram and metadata for a single song.

    Attributes:
        title (str): The title of the song.
        samples (list[np.ndarray]): A list of spectrogram fragments, where each fragment is a numpy array.
        tags (list[int]): Multi-hot encoded tags.
    """
    title: str
    samples: list[np.ndarray]
    tags: list[int]
