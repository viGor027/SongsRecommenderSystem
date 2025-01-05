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
            n_seconds: int | float,
            step: int | None = None
    ) -> tuple[list[np.ndarray], int] | tuple[None, None]:
        """
        This method loads an audio file from the given path, then divides it into equal-sized
        fragments, each with a duration of `n_seconds`.

        Notes:
            - Remainder at the end of the song is dropped.
            - By default, fragments are non-overlapping, with each fragment consisting of consecutive n_seconds of a song, specify `step` to change it.
            (when step is not specified it is sr*n_seconds)

        Args:
            path (str): The file path to the audio file to be fragmented.
            n_seconds (int): The duration of each fragment in seconds.
            step (int | float): Defines the interval (in seconds) at which consecutive fragments start within the audio.
                        If None fragments are non-overlapping.

        Returns:
            The method's return depends on whether the file was loaded successfully:
                - if loading a file caused no errors: A list containing the audio fragments as numpy arrays, the sample rate of the audio file.
                - if there was an error during loading a file: None, None.
        """
        try:
            song, sr = librosa.load(path)
            if step is None:
                step = int(n_seconds * sr)
            else:
                step = int(step*sr)

            return [song[i - int(n_seconds * sr):i] for i in range(int(n_seconds * sr), len(song), step)], sr
        except Exception as e:
            print(f"FeatureExtractor.make_fragments: Error when trying to load file from {path}")
            print(str(e))
            print(repr(e))
            FeatureExtractor.logger.append(path[len(os.path.dirname(path)):])
            return None, None


if __name__ == "__main__":
    # If you want to test the functionality of the class methods, do it here.

    # For example here is snippet comparing how many fragments will be there with step = 5 and with step=n_seconds=10
    paths = []
    fe = FeatureExtractor(paths)
    from song_pipeline.constants import SONGS_DIR
    path_to_song = os.path.join(SONGS_DIR, "A&B_-_ETikka__MADZI.mp3")
    print(len(fe.make_fragments(path_to_song, 10, 5)[0]))
    print(len(fe.make_fragments(path_to_song, 10)[0]))
    # n_seconds=10 and step=5 returns twice as many fragments than n_seconds=10 and step=10
