from workflow_actions.paths import (
    DATA_DIR,
    DOWNLOAD_DIR,
    FRAGMENTED_DATA_DIR,
    MODEL_READY_DATA_DIR,
)
from workflow_actions.dataset_preprocessor.source import (
    Chunker,
    RawAugment,
    SpectrogramExtractor,
    SpectrogramAugment,
    load_single_song_to_numpy,
    save_numpy_fragment,
    load_numpy_fragment,
    encode_song_labels_to_multi_hot_vector,
)
from workflow_actions.json_handlers import write_dict_to_json
from typing import TYPE_CHECKING, Literal
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import shutil
from datetime import datetime

if TYPE_CHECKING:
    from workflow_actions.dataset_preprocessor.source.chunker import FragmentedSongNumpy


@dataclass
class StartIndexes:
    start_index_train: int
    start_index_valid: int


@dataclass
class FragmentsIndex:
    """Maps global fragment number to song title that contains this fragment."""

    train_index: dict[int, str]
    valid_index: dict[int, str]

    def add_song_to_train(self, song_title: str, start_with_idx: int, n_fragments: int):
        for relative_idx in range(n_fragments):
            self.train_index[start_with_idx + relative_idx] = song_title

    def add_song_to_valid(self, song_title: str, start_with_idx: int, n_fragments: int):
        for relative_idx in range(n_fragments):
            self.valid_index[start_with_idx + relative_idx] = song_title

    def dump_indexes(self):
        write_dict_to_json(self.train_index, DATA_DIR / "train_index.json")
        write_dict_to_json(self.valid_index, DATA_DIR / "valid_index.json")


class DatasetPreprocessor:
    """
    prepare_all_songs_fragments:
        01_raw -> 02_fragmented (for both 'train' and 'valid'),
        fragments are stored as .npy files, labels are stored as .pt files,
        index files are created.

    make_fragments_model_ready_without_augmenting:
        02_fragmented -> 03_model_ready (for specified set),
        extracts spectrograms from fragments,
        y_<number>.pt files are just copied.

    pre_epoch_augment_hook:
        02_fragmented -> 03_model_ready (only 'train' set),
        applies augmentation on raw audio, extracts spectrograms, applies augmentation on spectrograms,
        y_<number>.pt files are just copied.

    For more info see respective functions docstrings.
    """

    def __init__(
        self,
        chunker: dict,
        raw_augment: dict,
        spectrogram_extractor: dict,
        spectrogram_augment: dict,
    ):
        self.chunker_cfg = chunker
        self.chunker = Chunker(**self.chunker_cfg)
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
        """
        For each song from 01_raw its labels and fragments are prepared
        and put into respective 02_fragmented directories.

        Fragments are saved as X_<number>.npy files,
        labels are encoded and saved as y_<number>.pt files.

        train_index.json and valid_index.json are created.

        Note: Function first empties train, valid directories of `02_fragmented`.
        """
        self._empty_folder(FRAGMENTED_DATA_DIR / "train")
        self._empty_folder(FRAGMENTED_DATA_DIR / "valid")
        start_indexes = StartIndexes(0, 0)
        fragments_index = FragmentsIndex(train_index={}, valid_index={})
        for song in DOWNLOAD_DIR.iterdir():
            if song.is_file():
                self._prepare_single_song_fragments(
                    song_title=song.name,
                    start_indexes=start_indexes,
                    fragments_index=fragments_index,
                )
        fragments_index.dump_indexes()
        self._create_fragmentation_stamp()

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
                self._make_single_fragment_model_ready(fragment_fname=fragment.name)

    def make_fragments_model_ready_without_augmenting(
        self,
        set_type: Literal["train", "valid"],
    ):
        """
        Use to move files from 02_fragmented to 03_model_ready without augmenting data.
        For every song fragment from 02_fragmented spectrogram is extracted and saved to 03_model_ready.

        set_type (str): Which set will be moved to 03_model_ready, either 'train' or 'valid'.

        Notes:
            - MODEL_READY_DATA_DIR / set_type is first emptied.
            - y_<number>.pt files are just copied.
        """
        self._empty_folder(MODEL_READY_DATA_DIR / set_type)
        for fragment_path in (FRAGMENTED_DATA_DIR / set_type).iterdir():
            if fragment_path.name.endswith(".npy"):
                loaded_numpy_fragment = load_numpy_fragment(path=fragment_path)
                spectrogram = self.spectrogram_extractor([loaded_numpy_fragment])[0]
                spectrogram_torch = torch.from_numpy(
                    spectrogram.astype(np.float32)
                ).unsqueeze(0)
                tensor_name = fragment_path.name.replace(".npy", ".pt")
                torch.save(
                    spectrogram_torch, MODEL_READY_DATA_DIR / set_type / tensor_name
                )
            else:
                shutil.copy(
                    fragment_path,
                    MODEL_READY_DATA_DIR / set_type / fragment_path.name,
                )

    def _prepare_single_song_fragments(
        self,
        song_title: str,
        start_indexes: StartIndexes,
        fragments_index: FragmentsIndex,
    ):
        """
        Creates song fragments, encodes its labels and saves them to 02_fragmented
        for train and valid sets.

        Note: **Pass `song_title` with extension.**
        """
        song, sample_rate = load_single_song_to_numpy(path=DOWNLOAD_DIR / song_title)
        fragmented_song: "FragmentedSongNumpy" = self.chunker.make_fragments_from_numpy(
            song=song
        )
        song_title = song_title.replace(".mp3", "")
        encoded_song_tags: torch.Tensor = encode_song_labels_to_multi_hot_vector(
            song_title=song_title
        )
        train_samples = fragmented_song["train"]
        n_train_samples = len(train_samples)
        self._save_set_fragments_with_labels(
            set_type="train",
            samples=train_samples,
            encoded_song_tags=encoded_song_tags,
            start_with_index=start_indexes.start_index_train,
        )
        fragments_index.add_song_to_train(
            song_title=song_title,
            start_with_idx=start_indexes.start_index_train,
            n_fragments=n_train_samples,
        )
        start_indexes.start_index_train += n_train_samples

        valid_samples = fragmented_song["valid"]
        n_valid_samples = len(valid_samples)
        self._save_set_fragments_with_labels(
            set_type="valid",
            samples=valid_samples,
            encoded_song_tags=encoded_song_tags,
            start_with_index=start_indexes.start_index_valid,
        )
        fragments_index.add_song_to_valid(
            song_title=song_title,
            start_with_idx=start_indexes.start_index_valid,
            n_fragments=n_valid_samples,
        )
        start_indexes.start_index_valid += n_valid_samples

    def _make_single_fragment_model_ready(self, fragment_fname: str):
        """
        Reads file `fragment_fname` from 02_fragmented/train,
        augments it and saves it into  03_model_ready/train.

        fragment_fname (str): fragment name like X_<number>.npy

        Note: y_<number>.pt files are just copied.
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

    def _create_fragmentation_stamp(self):
        now = datetime.now()
        formatted_time = now.strftime("%Y/%m/%d_%H-%M-%S")
        fragmentation_stamp = {**self.chunker_cfg, "time_stamp": formatted_time}
        write_dict_to_json(fragmentation_stamp, FRAGMENTED_DATA_DIR)

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
