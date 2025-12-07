import numpy as np
import torch

from .raw_augment import RawAugment
from .spectrogram_extractor import SpectrogramExtractor
from .spectrogram_augment import SpectrogramAugment


class FragmentPipeline:
    def __init__(
        self,
        apply_augmentations_on_raw: bool,
        extract_spectrograms: bool,
        apply_augmentations_on_spectrograms: bool,
        raw_augment: dict,
        spectrogram_extractor: dict,
        spectrogram_augment: dict,
    ):
        self.apply_augmentations_on_raw = apply_augmentations_on_raw
        self.extract_spectrograms = extract_spectrograms
        self.apply_augmentations_on_spectrograms = apply_augmentations_on_spectrograms

        self.raw_augment = RawAugment(**raw_augment)
        self.spectrogram_extractor = SpectrogramExtractor(**spectrogram_extractor)
        self.spectrogram_augment = SpectrogramAugment(**spectrogram_augment)

    def process_raw_fragments(
        self,
        fragments: list[np.ndarray],
        augment: bool = False,
    ):
        if self.apply_augmentations_on_raw and augment:
            fragments = self.raw_augment(
                fragments, sample_rate=self.spectrogram_extractor.sample_rate
            )

        if self.extract_spectrograms:
            fragments = self.spectrogram_extractor(fragments)

        fragments = [
            torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
            for x in fragments
        ]

        if (
            self.extract_spectrograms
            and self.apply_augmentations_on_spectrograms
            and augment
        ):
            fragments = self.spectrogram_augment(fragments)

        return fragments
