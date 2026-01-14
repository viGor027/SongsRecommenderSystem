from pathlib import Path
import numpy as np
import torch
import torchaudio
from typing import Optional
import librosa
from .label_encoder import LabelEncoder
from .global_fragments_index import GlobalFragmentsIndex
from workflow_actions.paths import (
    MODEL_READY_DATA_DIR,
    FRAGMENTATION_INDEX_PATH,
)
from workflow_actions.json_handlers import read_json_to_dict


class Serializer:
    def __init__(
        self,
        load_sample_rate: int,
        serialize_song_wise: bool,
        creating_fragmentation: bool,
    ):
        """
        Notes:
         - serialize_song_wise=False requires both indexes of GlobalFragmentsIndex to exist on disk.
         - serialize_song_wise=True stores one stacked X per song (X_<song_idx>.pt) and relies on a stable ordering of fragmentation_index keys.
        """
        self._load_sample_rate = load_sample_rate
        self._serialize_song_wise = serialize_song_wise

        self.label_encoder = LabelEncoder()
        self.global_fragments_index = GlobalFragmentsIndex()
        try:
            self.global_fragments_index.load_indexes()
            self._set_type_to_index = {
                "train": self.global_fragments_index.train_index,
                "valid": self.global_fragments_index.valid_index,
            }
        except FileNotFoundError as e:
            if not self._serialize_song_wise and not creating_fragmentation:
                print(
                    "Serializer.__init__ WARNING:\n"
                    "global_fragments indexes were not found on disk.\n"
                    "Only song-wise serialization is possible but serialize_song_wise was set to False.\n"
                    "Set serialize_song_wise to True and try again.\n"
                )
                raise e
        if not creating_fragmentation:
            self._fragmentation_index = read_json_to_dict(FRAGMENTATION_INDEX_PATH)[
                "fragmentation_index"
            ]

    def serialize_song_samples(
        self,
        samples: dict[str, list[torch.Tensor]],
        song_title,
        serialize_valid: bool,
    ):
        sets = ["train", "valid"] if serialize_valid else ["train"]
        for set_type in sets:
            if not self._serialize_song_wise:
                index = self._set_type_to_index[set_type]
                index_range = index[song_title]

                n_expected = index_range[1] - index_range[0] + 1
                if len(samples[set_type]) != n_expected:
                    raise RuntimeError(
                        f"{song_title} {set_type} expected {n_expected} samples, got {len(samples[set_type])}"
                    )
                for absolute_idx in range(index_range[0], index_range[1] + 1):
                    torch.save(
                        samples[set_type][absolute_idx - index_range[0]],
                        MODEL_READY_DATA_DIR / set_type / f"X_{absolute_idx}.pt",
                    )
            else:
                sample_idx = list(self._fragmentation_index.keys()).index(song_title)
                torch.save(
                    torch.stack(samples[set_type]),
                    MODEL_READY_DATA_DIR / set_type / f"X_{sample_idx}.pt",
                )

    def create_all_ys(self):
        for song_title in self._fragmentation_index.keys():
            self.create_single_song_ys(song_title=song_title)

    def create_single_song_ys(self, song_title):
        """
        Notes:
         - song_wise=True writes one y_<song_idx>.pt per song where song_idx is based on fragmentation_index key order.
         - song_wise=False writes y_<absolute_idx>.pt for each sample where absolute_idx is based on GlobalFragmentsIndex.
        """
        encoded_song_tags = self.label_encoder.encode_song_labels_to_multi_hot_vector(
            song_title=song_title,
        )
        for set_type in ["train", "valid"]:
            if self._serialize_song_wise:
                sample_idx = list(self._fragmentation_index.keys()).index(song_title)
                torch.save(
                    encoded_song_tags,
                    MODEL_READY_DATA_DIR / set_type / f"y_{sample_idx}.pt",
                )
            else:
                index = self._set_type_to_index[set_type]
                sample_range = index[song_title]
                for absolute_idx in range(sample_range[0], sample_range[1] + 1):
                    torch.save(
                        encoded_song_tags,
                        MODEL_READY_DATA_DIR / set_type / f"y_{absolute_idx}.pt",
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
                Serializer._resolve_audio_path(path), sr=self._load_sample_rate
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
