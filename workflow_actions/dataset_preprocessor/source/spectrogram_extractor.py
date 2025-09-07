from typing import TypeAlias, Literal, Callable
import librosa.feature
from typing import Iterable
import numpy as np

ExtractionMethod: TypeAlias = Literal["mel"]


class SpectrogramExtractor:
    def __init__(
        self, extraction_method: ExtractionMethod, sample_rate: int, n_mels: int
    ):
        extraction_methods: dict[ExtractionMethod, Callable] = {
            "mel": self._extract_mel_spec_from_fragments
        }

        self._extraction_method = extraction_methods[extraction_method]
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def _extract_mel_spec_from_fragments(
        self,
        fragments: Iterable[np.ndarray],
    ) -> list[np.ndarray]:
        """
        Extracts mel spectrograms from an iterable of audio fragments.

        Returns:
            list[np.ndarray]: A list of mel spectrograms in decibel scale, one for each audio fragment.
        """
        res = []
        for song_frag in fragments:
            spec = librosa.feature.melspectrogram(
                y=song_frag, sr=self.sample_rate, n_mels=self.n_mels
            )
            spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
            res.append(spec)
        return res

    def __call__(self, fragments: Iterable[np.ndarray]):
        return self._extraction_method(fragments)
