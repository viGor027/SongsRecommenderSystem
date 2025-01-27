import os.path
import numpy as np
import librosa
import librosa.display
import librosa.feature
from collections.abc import Iterable
from song_pipeline.dict_types import MelSpecKwargsType


class FeatureExtractor:
    """
    A utility class designed for extracting features from audio.

    Attributes:
        paths (Iterable[str]): An iterable containing the file paths of the audio files
            from which features are to be extracted.

    Args:
        paths (Iterable[str]): The file paths to the audio files to be processed.
    """
    logger = []

    def __init__(self, paths: Iterable[str]):
        self.paths = paths

    def extract_specs_from_paths(self) -> list[np.ndarray]:
        """
        Extracts spectrograms from the audio files specified in the `self.paths` attribute.

        Returns:
            list[np.ndarray]: A list of spectrograms in decibel scale, one for each audio file.
        """
        res = []
        for path in self.paths:
            song, sr = librosa.load(path)
            spec = librosa.stft(song)
            spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
            res.append(spec)
        return res

    def extract_mel_specs_from_paths(self, n_mels: int) -> list[np.ndarray]:
        """
        Extracts mel spectrograms from the audio files specified in the `self.paths` attribute.

        Args:
            n_mels (int): The number of mel bands to use in the mel spectrogram computation.

        Returns:
            list[np.ndarray]: A list of mel spectrograms in decibel scale, one for each audio file.
        """
        res = []
        for path in self.paths:
            song, sr = librosa.load(path)
            spec = librosa.feature.melspectrogram(
                y=song, sr=sr, n_mels=n_mels
            )
            spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
            res.append(spec)
        return res

    @staticmethod
    def extract_specs_from_fragments(fragments: Iterable[np.ndarray], **kwargs) -> list[np.ndarray]:
        """
        Extracts spectrograms from an iterable of audio fragments.

        Args:
            fragments (Iterable[np.ndarray]): An iterable containing audio fragments as numpy arrays.
            **kwargs: Additional keyword arguments included for consistency
                      with other methods. Not used in this function.
        Returns:
            list[np.ndarray]: A list of spectrograms in decibel scale, one for each audio fragment.
        """
        res = []
        for song_frag in fragments:
            spec = librosa.stft(song_frag)
            spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
            res.append(spec)
        return res

    @staticmethod
    def extract_mel_spec_from_fragments(
            fragments: Iterable[np.ndarray],
            **kwargs: MelSpecKwargsType
    ) -> list[np.ndarray]:
        """
        Extracts mel spectrograms from an iterable of audio fragments.

        Args:
            fragments (Iterable[np.ndarray]): An iterable containing audio fragments as numpy arrays.
            **kwargs (MelSpecKwargsType): Configuration options for mel spectrogram extraction, including:
                - sr (int): Sample rate of the audio fragments.
                - n_mels (int): The number of mel bands to use in the mel spectrogram computation.

        Returns:
            list[np.ndarray]: A list of mel spectrograms in decibel scale, one for each audio fragment.
        """
        res = []
        for song_frag in fragments:
            spec = librosa.feature.melspectrogram(
                y=song_frag, sr=kwargs['sr'], n_mels=kwargs['n_mels']
            )
            spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
            res.append(spec)
        return res

    @staticmethod
    def make_fragments(
            path: str,
            validation_probability: float,
            n_seconds: int | float,
            step: int | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], int] | tuple[None, None, None]:
        """
        This method loads an audio file from the given path, then divides it into equal-sized
        fragments, each with a duration of `n_seconds`.

        Notes:
            - Remainder at the end of the song is dropped.
            - By default, fragments are non-overlapping, with each fragment consisting of consecutive n_seconds of a song, specify `step` to change it.
            (when step is not specified it is sr*n_seconds)

        Args:
            path (str): The file path to the audio file to be fragmented.
            validation_probability (float): Defines a chance of fragment being added to a validation set;
                        fragment gets to a training set with a chance equal to 1-validation_probability.
            n_seconds (int): The duration of each fragment in seconds.
            step (int | float): Defines the interval (in seconds) at which consecutive fragments start within the audio.
                        If None fragments are non-overlapping.

        Returns:
            The method's return depends on whether the file was loaded successfully:
                - if loading a file caused no errors: A tuple (training fragments, validation fragments, sample rate).
                - if there was an error during loading a file: A tuple (None, None, None).
        """
        if not (0 <= validation_probability <= 1):
            raise ValueError(f"Number {validation_probability} is out of range. Expected range is 0 to 1, including 0 "
                             f"and 1.")

        try:
            song, sr = librosa.load(path)
            if step is None:
                step = int(n_seconds * sr)
            else:
                step = int(step * sr)
            valid = []
            train = []
            i = int(n_seconds * sr)
            recent_sample_to_valid = None
            while i < len(song):
                sample_to_valid = np.random.choice([False, True],
                                                   p=[1 - validation_probability, validation_probability])
                if sample_to_valid and recent_sample_to_valid:
                    # current sample goes to validation set, recent sample also went to validation set
                    # no need for skipping
                    valid.append(song[i - int(n_seconds * sr):i])
                    recent_sample_to_valid = True
                elif sample_to_valid and not recent_sample_to_valid:
                    # current sample goes to validation set, recent sample went to training set,
                    # we skip the part that overlap
                    i += int(n_seconds * sr) - step
                    if i >= len(song):
                        break
                    valid.append(song[i - int(n_seconds * sr):i])
                    recent_sample_to_valid = True
                elif not sample_to_valid and not recent_sample_to_valid:
                    # current sample goes to training set, recent sample went to train,
                    # no need for skipping
                    train.append(song[i - int(n_seconds * sr):i])
                    recent_sample_to_valid = False
                else:
                    # current sample goes to validation set, recent sample went to training set,
                    # we skip the part that overlap
                    i += int(n_seconds * sr) - step
                    if i >= len(song):
                        break
                    valid.append(song[i - int(n_seconds * sr):i])
                    recent_sample_to_valid = False
                i += step
            return train, valid, sr
        except Exception as e:
            print(f"FeatureExtractor.make_fragments: Error when trying to load file from {path}")
            print(str(e))
            print(repr(e))
            FeatureExtractor.logger.append(path[len(os.path.dirname(path)):])
            return None, None, None
