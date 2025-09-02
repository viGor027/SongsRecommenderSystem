from workflow_actions.paths import (
    LABELS_PATH,
    LABEL_MAPPING_PATH,
    DOWNLOAD_DIR,
    FRAGMENTED_DATA_DIR,
    MODEL_READY_DATA_DIR,
)
from workflow_actions.prepare_dataset.source.raw_augment import RawAugment
from workflow_actions.prepare_dataset.source.serializer import (
    load_single_song_to_numpy,
    save_numpy_fragment,
    load_numpy_fragment,
)
from workflow_actions.prepare_dataset.source.label_encoder import (
    encode_song_labels_to_multi_hot_vector,
)
from workflow_actions.prepare_dataset.source.spectrogram_extractor import (
    SpectrogramExtractor,
)
from workflow_actions.prepare_dataset.source.spectrogram_augment import (
    SpectrogramAugment,
)
from workflow_actions.prepare_dataset.source.chunker import Chunker
from workflow_actions.json_handlers import write_dict_to_json, read_json_to_dict
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
        spectrogram_augment: dict,
    ):
        self.label_mapping = self.create_label_mapping()
        self.chunker = Chunker(**chunker)
        self.raw_augment = RawAugment(**raw_augment)
        self.spectrogram_extractor = SpectrogramExtractor(**spectrogram_extractor)
        self.spectrogram_augment = SpectrogramAugment(**spectrogram_augment)

        if self.chunker.sample_rate != self.spectrogram_extractor.sample_rate:
            raise ValueError(
                (
                    "Chunker sample rate must match SpectrogramExtractor sample rate. ",
                    "Check prepare_dataset_config.json values.",
                )
            )

    def prepare_all_songs_fragments(self):
        """Function first empties train, valid directories of `02_fragmented`."""
        self._empty_folder(FRAGMENTED_DATA_DIR / "train")
        self._empty_folder(FRAGMENTED_DATA_DIR / "valid")
        start_indexes = PrepareDatasetStartIndexes(0, 0)
        for song in DOWNLOAD_DIR.iterdir():
            if song.is_file():
                self.prepare_single_song_fragments(
                    song_title=song.name, start_indexes=start_indexes
                )

    def prepare_single_song_fragments(
        self, song_title: str, start_indexes: PrepareDatasetStartIndexes
    ):
        """
        Creates song fragments with labels and saves them to 02_fragmented.

        **Pass `song_title` with extension.**

        start_w_index_train/valid (int):
            The first number we should start saving current song fragments with.

        """
        song, sample_rate = load_single_song_to_numpy(path=DOWNLOAD_DIR / song_title)
        fragmented_song: "FragmentedSongNumpy" = self.chunker.make_fragments_from_numpy(
            song=song
        )
        encoded_song_tags: torch.Tensor = encode_song_labels_to_multi_hot_vector(
            song_title=song_title.replace(".mp3", "")
        )
        train_samples = fragmented_song["train"]
        self._save_set_fragments_with_labels(
            set_type="train",
            samples=train_samples,
            encoded_song_tags=encoded_song_tags,
            start_with_index=start_indexes.start_index_train,
        )
        start_indexes.start_index_train += len(train_samples)

        valid_samples = fragmented_song["valid"]
        self._save_set_fragments_with_labels(
            set_type="valid",
            samples=valid_samples,
            encoded_song_tags=encoded_song_tags,
            start_with_index=start_indexes.start_index_valid,
        )
        start_indexes.start_index_valid += len(valid_samples)

    def pre_epoch_augment_hook(self):
        """
        Is called at the beginning of every epoch;
        takes fragments from 02_fragmented/train,
        applies augmentations on raw fragment,
        extracts mel spectrogram,
        applies augmentations on mel spectrograms,
        saves spectrogram fragments to 03_model_ready/train.

        Note:
            Function empties 03_model_ready/train directory
            at the beginning of each call.
        """
        # TODO: Make it parallel
        self._empty_folder(MODEL_READY_DATA_DIR / "train")
        for fragment in (FRAGMENTED_DATA_DIR / "train").iterdir():
            if (
                fragment.is_file()
                and fragment.name.startswith("X_")
                and fragment.name.endswith(".npy")
            ):
                self.make_single_fragment_model_ready(fragment_fname=fragment.name)

    def make_single_fragment_model_ready(self, fragment_fname: str):
        """
        Reads file `fragment_fname` from 02_fragmented/train,
        augments it and saves it into  03_model_ready/train.

        fragment_fname (str): fragment name like X_<number>.npy
        """
        if not all([substr in fragment_fname for substr in ["X", "_", ".npy"]]):
            raise ValueError(
                "Wrong fragment_fname passed to "
                f"PrepareDataset.make_single_fragment_model_ready: {fragment_fname}"
            )

        fragment = load_numpy_fragment(
            path=FRAGMENTED_DATA_DIR / "train" / fragment_fname
        )
        fragment_id: str = fragment_fname.split("_")[1].split(".")[0]
        augmented_raw_fragment, _ = self.raw_augment(
            fragment_id=fragment_id,
            raw_fragment=fragment,
            sample_rate=self.chunker.sample_rate,
        )
        fragment_mel_spec = self.spectrogram_extractor([augmented_raw_fragment])[0]
        augmented_mel_spec, _ = self.spectrogram_augment(
            fragment_id=fragment_id, spectrogram=fragment_mel_spec
        )
        augmented_mel_spec_fname = f"X_{fragment_id}.pt"
        torch.save(
            augmented_mel_spec,
            MODEL_READY_DATA_DIR / "train" / augmented_mel_spec_fname,
        )
        y_fname = f"y_{fragment_id}.pt"
        shutil.copy(
            FRAGMENTED_DATA_DIR / "train" / y_fname,
            MODEL_READY_DATA_DIR / "train" / y_fname,
        )

    @staticmethod
    def make_set_content_model_ready(set_type: Literal["train", "valid"]):
        """
        Use to move files from 02_fragmented to 03_model_ready without augmenting data.

        Notes:
            - Function converts .npy files from 02_fragmented to .pt files.
            - X_<number>.npy files are converted to .pt files
            - y_<number>.pt files are just copied.
        """
        for fragment_path in (FRAGMENTED_DATA_DIR / set_type).iterdir():
            if fragment_path.name.endswith(".npy"):
                loaded_numpy_file = load_numpy_fragment(path=fragment_path)
                tensor = torch.from_numpy(loaded_numpy_file)
                tensor_name = fragment_path.name.replace(".npy", ".pt")
                torch.save(tensor, MODEL_READY_DATA_DIR / set_type / tensor_name)
            else:
                shutil.copy(
                    fragment_path,
                    MODEL_READY_DATA_DIR / set_type / fragment_path.name,
                )

    @staticmethod
    def _empty_folder(path: Path):
        """Removes everything in `path` folder"""
        for element in path.iterdir():
            if element.is_file():
                element.unlink()
            elif element.is_dir():
                shutil.rmtree(element)

    @staticmethod
    def _save_set_fragments_with_labels(
        set_type: Literal["train", "valid"],
        samples: list,
        encoded_song_tags: torch.Tensor,
        start_with_index: int,
    ):
        for current_relative_idx, fragment in enumerate(samples):
            absolute_fragment_number = current_relative_idx + start_with_index
            print("DEBUG _save_set_fragments_with_labels: ", fragment.shape)
            save_numpy_fragment(
                fragment=fragment,
                path=FRAGMENTED_DATA_DIR
                / set_type
                / f"X_{absolute_fragment_number}.npy",
            )
            torch.save(
                encoded_song_tags,
                FRAGMENTED_DATA_DIR / set_type / f"y_{absolute_fragment_number}.pt",
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
