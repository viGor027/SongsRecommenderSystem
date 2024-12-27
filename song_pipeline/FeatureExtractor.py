import numpy as np
import librosa
import librosa.display
import librosa.feature
import os
from typing import List, Tuple
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

    def __init__(self, paths: Iterable[str]):
        self.paths = paths

    def extract_specs_from_paths(self) -> List[np.ndarray]:
        """
        Extracts spectrograms from the audio files specified in the `paths` attribute.

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

    def extract_mel_specs_from_paths(self, n_mels: int) -> List[np.ndarray]:
        """
        Extracts mel spectrograms from the audio files specified in the `paths` attribute.

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
    def extract_specs_from_fragments(fragments: Iterable[np.ndarray], **kwargs) -> List[np.ndarray]:
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
    ) -> List[np.ndarray]:
        """
        Extracts mel spectrograms from an iterable of audio fragments.

        Args:
            fragments (Iterable[np.ndarray]): An iterable containing audio fragments as numpy arrays.
            **kwargs (MelSpecKwargs): Configuration options for mel spectrogram extraction, including:
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
    def make_fragments(path: str, n_seconds: int, step: int | None = None) -> Tuple[List[np.ndarray], int]:
        """
        This method loads an audio file from the given path, then divides it into equal-sized
        fragments, each with a duration of `n_seconds`. The fragments are returned as a list of
        numpy arrays.

        Note: Remainder at the end of the song is dropped.

        Args:
            path (str): The file path to the audio file to be fragmented.
            n_seconds (int): The duration of each fragment in seconds.
            step (int): Defines the interval at which consecutive fragments start within the audio.
                        If None fragments are non-overlapping.

        Returns:
            tuple:
                list[np.ndarray]: A list containing the audio fragments as numpy arrays.
                int: The sample rate of the audio file.
        """
        song, sr = librosa.load(path)

        if step is None:
            step = n_seconds * sr

        return [song[i - n_seconds * sr:i] for i in range(n_seconds * sr, len(song), step)], sr


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'sample',
        'Eminem - Business (Matoma Remix).mp3'
    )

    paths = [sample_path]
    # print("#1")
    song, sr = librosa.load(sample_path)
    # print(type(song), type(sr))
    # print("#2")
    print(song.shape, sr, song.shape[0] // sr)  # ilosc_probek, sample_rate, długosc_piosenki_w_sekundach

    # print("feature", __name__, __file__)  # sanity check
    fe = FeatureExtractor(paths)
    spec = fe.extract_mel_specs_from_paths(n_mels=100)[0]
    print(spec.shape, type(spec))
    fig, ax = plt.subplots(figsize=(14, 7))
    img = librosa.display.specshow(spec, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax)
    plt.show()
