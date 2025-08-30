import numpy as np
import torch
from typing import TypedDict, Literal
from numpy.typing import NDArray


class FragmentedSongNumpy(TypedDict):
    train: list[NDArray[np.float32]]
    valid: list[NDArray[np.float32]]
    sample_rate: int


class FragmentedSongTorch(TypedDict):
    train: list[torch.Tensor]
    valid: list[torch.Tensor]
    sample_rate: int


class Chunker:
    """
    Divides single song into equal-sized fragments, each with a duration of `n_seconds`.

    Notes:
        - Remainder at the end of the song is dropped.
        - By default, fragments are non-overlapping, with each fragment consisting of consecutive n_seconds of a song,
        specify `step` to change it. (when step is not specified it is sr*n_seconds)
    """
    def __init__(self,
                 song_type: Literal['torch', 'numpy'],
                 sample_rate: int,
                 validation_probability: float,
                 n_seconds: int | float,
                 step: int | float | None = None):
        """
        Args:
            song_type (Literal['torch', 'numpy']): What will be the song type chunker will be called with.
            sample_rate (int): Sample rate of a song.
            validation_probability (float): Defines a chance of fragment being added to a validation set;
                        fragment gets to a training set with a chance equal to 1-validation_probability.
            n_seconds (int): The duration of each fragment in seconds.
            step (int | float | None):
                Defines the interval (in seconds) at which consecutive fragments start within the audio.
                If None fragments are non-overlapping.
        """
        self.sample_rate = sample_rate
        self.validation_probability = validation_probability
        self.n_seconds = n_seconds
        self.step = step

        chunker_map = {
            'torch': self.make_fragments_from_torch,
            'numpy': self.make_fragments_from_numpy
        }
        self._call_func = chunker_map[song_type]

    def __call__(self, song: torch.Tensor | NDArray[np.float32]) -> FragmentedSongTorch | FragmentedSongNumpy:
        return self._call_func(song)

    def make_fragments_from_numpy(self, song: NDArray[np.float32]) -> FragmentedSongNumpy | None:
        if not (0 <= self.validation_probability <= 1):
            raise ValueError(f"Number {self.validation_probability} is out of range."
                             f"Expected range is 0 to 1, including 0 and 1.")
        try:
            song, sr = song, self.sample_rate
            if self.step is None:
                step = int(self.n_seconds * sr)
            else:
                step = int(self.step * sr)
            valid = []
            train = []
            i = int(self.n_seconds * sr)
            recent_sample_to_valid = None
            while i < len(song):
                sample_to_valid = np.random.choice([False, True],
                                                   p=[1 - self.validation_probability, self.validation_probability])
                if sample_to_valid and recent_sample_to_valid:
                    # current sample goes to validation set, recent sample also went to validation set
                    # no need for skipping
                    valid.append(song[i - int(self.n_seconds * sr):i].copy())
                    recent_sample_to_valid = True
                elif sample_to_valid and not recent_sample_to_valid:
                    # current sample goes to validation set, recent sample went to training set,
                    # we skip the part that overlap
                    i += int(self.n_seconds * sr) - step
                    if i >= len(song):
                        break
                    valid.append(song[i - int(self.n_seconds * sr):i].copy())
                    recent_sample_to_valid = True
                elif not sample_to_valid and not recent_sample_to_valid:
                    # current sample goes to training set, recent sample went to train,
                    # no need for skipping
                    train.append(song[i - int(self.n_seconds * sr):i].copy())
                    recent_sample_to_valid = False
                else:
                    # current sample goes to validation set, recent sample went to training set,
                    # we skip the part that overlap
                    i += int(self.n_seconds * sr) - step
                    if i >= len(song):
                        break
                    valid.append(song[i - int(self.n_seconds * sr):i].copy())
                    recent_sample_to_valid = False
                i += step
            return FragmentedSongNumpy(train=train, valid=valid, sample_rate=sr)
        except Exception as e:
            print(f"make_fragments: Error when trying to fragment song. {str(e)}")
            return None

    def make_fragments_from_torch(self, song: torch.Tensor) -> FragmentedSongTorch | None:
        if not (0 <= self.validation_probability <= 1):
            raise ValueError(f"Number {self.validation_probability} is out of range. "
                             f"Expected range is 0 to 1, including 0 and 1.")
        try:
            song, sr = song, self.sample_rate
            if self.step is None:
                step = int(self.n_seconds * sr)
            else:
                step = int(self.step * sr)
            valid = []
            train = []
            i = int(self.n_seconds * sr)
            recent_sample_to_valid = None
            while i < song.shape[-1]:
                sample_to_valid = np.random.choice([False, True],
                                                   p=[1 - self.validation_probability, self.validation_probability])
                if sample_to_valid and recent_sample_to_valid:
                    # current sample goes to validation set, recent sample also went to validation set
                    # no need for skipping
                    valid.append(song[..., i - int(self.n_seconds * sr):i].contiguous().clone())
                    recent_sample_to_valid = True
                elif sample_to_valid and not recent_sample_to_valid:
                    # current sample goes to validation set, recent sample went to training set,
                    # we skip the part that overlap
                    i += int(self.n_seconds * sr) - step
                    if i >= song.shape[-1]:
                        break
                    valid.append(song[..., i - int(self.n_seconds * sr):i].contiguous().clone())
                    recent_sample_to_valid = True
                elif not sample_to_valid and not recent_sample_to_valid:
                    # current sample goes to training set, recent sample went to train,
                    # no need for skipping
                    train.append(song[..., i - int(self.n_seconds * sr):i].contiguous().clone())
                    recent_sample_to_valid = False
                else:
                    # current sample goes to validation set, recent sample went to training set,
                    # we skip the part that overlap
                    i += int(self.n_seconds * sr) - step
                    if i >= song.shape[-1]:
                        break
                    train.append(song[..., i - int(self.n_seconds * sr):i].contiguous().clone())
                    recent_sample_to_valid = False
                i += step
            return FragmentedSongTorch(train=train, valid=valid, sample_rate=sr)
        except Exception as e:
            print(f"make_fragments: Error when trying to fragment song. {str(e)}")
            return None
