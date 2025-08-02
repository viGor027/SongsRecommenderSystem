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
        validation_probability (float): Defines a chance of fragment being added to a validation set;
                        fragment gets to a training set with a chance equal to 1-validation_probability.
        spec_type (Literal['mel', 'std']): The type of spectrogram being used.
                - `'mel'`: Mel spectrogram.
                - `'std'`: Standard spectrogram.
    """
    n_mels: int
    n_seconds: int
    step: int | float
    validation_probability: float
    spec_type: Literal['mel', 'std']


class SongSpecDataType(TypedDict):
    """
    Represents the spectrogram and metadata for a single song.

    Attributes:
        title (str): The title of the song.
        samples (list[np.ndarray]): A list of spectrogram fragments.
        tags (list[int]): Multi-hot encoded tags.
    """
    title: str
    samples: list[np.ndarray]
    tags: list[int]
