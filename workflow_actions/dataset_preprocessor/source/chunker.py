import numpy as np
from typing import TypedDict
from numpy.typing import NDArray


class FragmentedSongSlices(TypedDict):
    train: list[NDArray[np.float32]]
    valid: list[NDArray[np.float32]]


class FragmentedSongIndex(TypedDict):
    train: list[list[int, int]]
    valid: list[list[int, int]]


class Chunker:
    """
    Divides single song into equal-sized fragments, each with a duration of `n_seconds`.

    Notes:
        - Remainder at the end of the song is dropped.
        - By default, fragments are non-overlapping, with each fragment consisting of consecutive n_seconds of a song,
        specify `step` to change it. (when step is not specified it is sr*n_seconds)
    """

    def __init__(
        self,
        sample_rate: int,
        validation_probability: float,
        n_seconds: int | float,
        step: int | float | None = None,
    ):
        """
        Args:
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
        if step is not None and step >= n_seconds:
            raise ValueError("Chunker it must be that step < n_seconds.")

    def make_song_index(
        self,
        song: NDArray[np.float32],
    ) -> FragmentedSongIndex:
        return self._fragmentation_core(song=song)

    @staticmethod
    def make_song_slices(
        song: NDArray[np.float32], fragmented_song_index: FragmentedSongIndex
    ) -> FragmentedSongSlices:
        train_slices = [
            Chunker._get_slice(song=song, left=left, right=right)
            for (left, right) in fragmented_song_index["train"]
        ]
        valid_slices = [
            Chunker._get_slice(song=song, left=left, right=right)
            for (left, right) in fragmented_song_index["valid"]
        ]
        return FragmentedSongSlices(
            train=train_slices,
            valid=valid_slices,
        )

    def _fragmentation_core(
        self,
        song: NDArray[np.float32],
    ) -> FragmentedSongIndex:
        if not (0 <= self.validation_probability <= 1):
            raise ValueError(
                f"Number {self.validation_probability} is out of range."
                f"Expected range is 0 to 1, including 0 and 1."
            )
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
                sample_to_valid = np.random.choice(
                    [False, True],
                    p=[1 - self.validation_probability, self.validation_probability],
                )
                if sample_to_valid and (
                    recent_sample_to_valid is None or recent_sample_to_valid
                ):
                    # current sample goes to validation set, recent sample also went to validation set
                    # no need for skipping
                    valid.append([i - int(self.n_seconds * sr), i])
                    recent_sample_to_valid = True
                elif sample_to_valid and not recent_sample_to_valid:
                    # current sample goes to validation set, recent sample went to training set,
                    # we skip the part that overlap
                    i += int(self.n_seconds * sr) - step
                    if i >= len(song):
                        break
                    valid.append([i - int(self.n_seconds * sr), i])
                    recent_sample_to_valid = True
                elif not sample_to_valid and (
                    recent_sample_to_valid is None or not recent_sample_to_valid
                ):
                    # current sample goes to training set, recent sample went to train,
                    # no need for skipping
                    train.append([i - int(self.n_seconds * sr), i])
                    recent_sample_to_valid = False
                else:
                    # current sample goes to train set, recent sample went to validation set,
                    # we skip the part that overlap
                    i += int(self.n_seconds * sr) - step
                    if i >= len(song):
                        break
                    train.append([i - int(self.n_seconds * sr), i])
                    recent_sample_to_valid = False
                i += step
            return FragmentedSongIndex(train=train, valid=valid)
        except Exception as e:
            print(f"_fragmentation_core: Error when trying to fragment song. {str(e)}")

    @staticmethod
    def _get_slice(
        song: NDArray[np.float32],
        left: int,
        right: int,
    ) -> NDArray[np.float32]:
        return song[left:right].copy()
