from pathlib import Path
import numpy as np
import torch
import torchaudio
from typing import Optional
import librosa
from .label_encoder import LabelEncoder
from .global_fragments_index import GlobalFragmentsIndex
from workflow_actions.paths import (
    MODEL_READY_TRAIN_DIR,
    MODEL_READY_VALID_DIR,
)


class Serializer:
    def __init__(self, load_sample_rate: int):
        self.load_sample_rate = load_sample_rate
        self.label_encoder = LabelEncoder()
        self.global_fragments_index = GlobalFragmentsIndex()
        self.global_fragments_index.load_indexes()
        self._path_and_index_for_set_type = {
            "train": (MODEL_READY_TRAIN_DIR, self.global_fragments_index.train_index),
            "valid": (MODEL_READY_VALID_DIR, self.global_fragments_index.valid_index),
        }

    def serialize_song_samples(
        self,
        samples: dict[str, list[torch.Tensor]],
        song_title,
        serialize_valid: bool,
    ):
        sets = ["train", "valid"] if serialize_valid else ["train"]
        for set_type in sets:
            set_path, index = self._path_and_index_for_set_type[set_type]
            index_range = index[song_title]

            n_expected = index_range[1] - index_range[0] + 1
            if len(samples[set_type]) != n_expected:
                raise RuntimeError(
                    f"{song_title} {set_type} expected {n_expected} samples, got {len(samples[set_type])}"
                )

            for absolute_idx in range(index_range[0], index_range[1] + 1):
                torch.save(
                    samples[set_type][absolute_idx - index_range[0]],
                    set_path / f"X_{absolute_idx}.pt",
                )

    def create_all_ys(self):
        for (
            set_path,
            index,
        ) in self._path_and_index_for_set_type.values():
            for song_title, sample_range in index.items():
                encoded_song_tags = (
                    self.label_encoder.encode_song_labels_to_multi_hot_vector(
                        song_title=song_title,
                    )
                )
                for absolute_idx in range(sample_range[0], sample_range[1] + 1):
                    torch.save(
                        encoded_song_tags,
                        set_path / f"y_{absolute_idx}.pt",
                    )

    def create_single_song_ys(self, song_title):
        for (
            set_path,
            index,
        ) in self._path_and_index_for_set_type.values():
            encoded_song_tags = (
                self.label_encoder.encode_song_labels_to_multi_hot_vector(
                    song_title=song_title,
                )
            )
            sample_range = index[song_title]
            for absolute_idx in range(sample_range[0], sample_range[1] + 1):
                torch.save(
                    encoded_song_tags,
                    set_path / f"y_{absolute_idx}.pt",
                )

    @staticmethod
    def save_numpy_fragment(fragment: np.ndarray, path: Path):
        np.save(str(path), fragment)

    @staticmethod
    def load_numpy_fragment(path: Path) -> np.ndarray:
        return np.load(str(path))

    def load_single_song_to_numpy(
        self,
        path: Path,
    ) -> tuple[np.ndarray, int] | tuple[None, None]:
        try:
            song, sample_rate = librosa.load(
                Serializer._resolve_audio_path(path), sr=self.load_sample_rate
            )
        except Exception as _:
            print(f"There was a problem loading {path.stem}; Skipping...")
            return None, None
        return song, sample_rate

    @staticmethod
    def load_single_song_to_torch(
        path: Path, new_sample_rate: Optional[int]
    ) -> tuple[torch.Tensor, int]:
        wave, sr = torchaudio.load(path)
        wave_mono = wave.mean(dim=0, keepdim=False)
        if new_sample_rate:
            wave_resampled = torchaudio.functional.resample(
                wave_mono, orig_freq=sr, new_freq=new_sample_rate
            )
            return wave_resampled, new_sample_rate
        return wave_mono, sr

    @staticmethod
    def _resolve_audio_path(path: Path) -> Path:
        base = path.with_suffix("")
        for ext in (".wav", ".mp3"):
            candidate = base.with_suffix(ext)
            if candidate.exists():
                return candidate
