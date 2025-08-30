from workflow_actions.paths import (
    LABELS_PATH,
    LABEL_MAPPING_PATH,
    DOWNLOAD_DIR,
    FRAGMENTED_DATA_DIR,
    MODEL_READY_DATA_DIR
)
from workflow_actions.json_handlers import write_dict_to_json
from raw_augment import RawAugment
from serializer import load_single_song_to_numpy, save_numpy_fragment, load_numpy_fragment
from label_encoder import encode_song_labels_to_multi_hot_vector
from spectrogram_extractor import SpectrogramExtractor
from spectrogram_augment import SpectrogramAugment
from chunker import Chunker
from collections import Counter
from typing import TYPE_CHECKING, Literal
import torch
from dataclasses import dataclass
from pathlib import Path
import shutil

if TYPE_CHECKING:
    from chunker import FragmentedSongNumpy


@dataclass
class PrepareDatasetStartIndexes:
    start_index_train: int
    start_index_valid: int


class PrepareDataset:
    def __init__(
            self,
            chunker: dict,
            raw_augment: dict,
            spectrogram_extractor: dict,
            spectrogram_augment: dict
    ):
        self.label_mapping = self.create_label_mapping()
        self.chunker = Chunker(**chunker)
        self.raw_augment = RawAugment(**raw_augment)
        self.spectrogram_extractor = SpectrogramExtractor(**spectrogram_extractor)
        self.spectrogram_augment = SpectrogramAugment(**spectrogram_augment)

        if self.chunker.sample_rate != self.spectrogram_extractor.sample_rate:
            raise ValueError((
                "Chunker sample rate must match SpectrogramExtractor sample rate. ",
                "Check prepare_dataset_config.json values."
            ))

    def prepare_all_songs_fragments(self):
        """Function first empties train, valid directories of `02_fragmented`."""
        self._empty_folder(FRAGMENTED_DATA_DIR / "train")
        self._empty_folder(FRAGMENTED_DATA_DIR / "valid")
        start_indexes = PrepareDatasetStartIndexes(0, 0)
        for song in DOWNLOAD_DIR.iterdir():
            if song.is_file():
                self.prepare_single_song_fragments(
                    song_title=song.name,
                    start_indexes=start_indexes
                )

    def prepare_single_song_fragments(
            self,
            song_title: str,
            start_indexes: PrepareDatasetStartIndexes
    ):
        """
        Creates song fragments with labels and saves them to 02_fragmented.

        **Pass `song_title` with extension.**

        start_w_index_train/valid (int):
            The first number we should start saving current song fragments with.

        """
        song, sample_rate = load_single_song_to_numpy(
            path=DOWNLOAD_DIR / song_title
        )
        fragmented_song: 'FragmentedSongNumpy' = self.chunker.make_fragments_from_numpy(song=song)
        encoded_song_tags: torch.Tensor = encode_song_labels_to_multi_hot_vector(
            song_title=song_title.replace(".mp3", "")
        )
        train_samples = fragmented_song['train']
        self._save_set_fragments_with_labels(
            set_type='train',
            samples=train_samples,
            encoded_song_tags=encoded_song_tags,
            start_with_index=start_indexes.start_index_train
        )
        start_indexes.start_index_train += len(train_samples)

        valid_samples = fragmented_song['valid']
        self._save_set_fragments_with_labels(
            set_type='valid',
            samples=valid_samples,
            encoded_song_tags=encoded_song_tags,
            start_with_index=start_indexes.start_index_valid
        )
        start_indexes.start_index_valid += len(valid_samples)

    def pre_epoch_augment_hook(self):
        """
        Is called at the beginning of every epoch;
        takes fragments from 02_fragmented,
        applies augmentations on raw fragment,
        extracts mel spectrogram,
        applies augmentations on mel spectrograms,
        saves spectrogram fragments to 03_model_ready.

        Note: Function empties
        """
        pass

    def make_single_fragment_model_ready(
            self,
            fragment_fname: str,
            set_type: Literal['train', 'valid'],
    ):
        """
        fragment_fname (str): fragment name like X_<number>.npy
        """
        fragment = load_numpy_fragment(
            path=FRAGMENTED_DATA_DIR / set_type / fragment_fname
        )
        fragment_id: str = fragment_fname.split("_")[1].split(".")[0]
        augmented_raw_fragment, _ = self.raw_augment(
            fragment_id=fragment_id,
            raw_fragment=fragment,
            sample_rate=self.chunker.sample_rate
        )
        fragment_mel_spec = self.spectrogram_extractor([augmented_raw_fragment])[0]
        augmented_mel_spec, _ = self.spectrogram_augment(
            fragment_id=fragment_id,
            spectrogram=fragment_mel_spec
        )
        augmented_mel_spec_fname = f"X_{fragment_id}.pt"
        torch.save(
            augmented_mel_spec,
            MODEL_READY_DATA_DIR / set_type / augmented_mel_spec_fname
        )
        y_fname = f"y_{fragment_id}.pt"
        shutil.copy(
            FRAGMENTED_DATA_DIR / set_type / y_fname,
            MODEL_READY_DATA_DIR / set_type / y_fname
        )

    @staticmethod
    def _empty_folder(path: Path):
        """Removes only files, leaves directories untouched."""
        for element in path.iterdir():
            if element.is_file():
                element.unlink()

    @staticmethod
    def _save_set_fragments_with_labels(
            set_type: Literal['train', 'valid'],
            samples: list,
            encoded_song_tags: torch.Tensor,
            start_with_index: int
    ):
        for current_relative_idx, fragment in enumerate(samples):
            absolute_fragment_number = current_relative_idx + start_with_index
            print("DEBUG _save_set_fragments_with_labels: ", fragment.shape)
            save_numpy_fragment(
                fragment=fragment,
                path=FRAGMENTED_DATA_DIR / set_type / f"X_{absolute_fragment_number}.npy"
            )
            torch.save(
                encoded_song_tags,
                FRAGMENTED_DATA_DIR / set_type / f"y_{absolute_fragment_number}.pt"
            )

    @staticmethod
    def create_label_mapping() -> dict[int, str]:
        """Creates mapping tag -> number for every tag present in labels.json"""
        if not LABELS_PATH.exists():
            raise FileExistsError("Labels file labels.json doesn't exist.")

        song_to_labels = read_json_to_dict(LABELS_PATH)
        all_tags = []
        for tags_list in song_to_labels.values():
            all_tags.extend(tags_list)
        tags = Counter(all_tags).keys()

        mapping = {label: idx for idx, label in enumerate(sorted(list(tags)))}
        write_dict_to_json(data=mapping, file_path=LABEL_MAPPING_PATH)
        return mapping


if __name__ == "__main__":
    from pathlib import Path
    from workflow_actions.json_handlers import read_json_to_dict

    config_path = "D:\\Nauka\\Projekty\\SongsRecommenderSystem\\workflow_actions\\prepare_dataset\\prepare_dataset_config.json"
    chunker_pd_config = read_json_to_dict(config_path)['chunker']
    raw_augment_config = read_json_to_dict(config_path)['raw_augment']
    spectrogram_extractor = read_json_to_dict(config_path)['spectrogram_extractor']
    spectrogram_augment = read_json_to_dict(config_path)['spectrogram_augment']
    pd = PrepareDataset(
        chunker=chunker_pd_config,
        raw_augment=raw_augment_config,
        spectrogram_extractor=spectrogram_extractor,
        spectrogram_augment=spectrogram_augment
    )
    pd.make_single_fragment_model_ready(
        fragment_fname="X_0.npy",
        set_type="valid"
    )
