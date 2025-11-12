from workflow_actions.paths import (
    DOWNLOAD_DIR,
    FRAGMENTATION_STAMP_PATH,
    FRAGMENTATION_INDEX_PATH,
    SCRAPE_STAMP_PATH,
    GLOBAL_TRAIN_INDEX_PATH,
    GLOBAL_VALID_INDEX_PATH,
    MODEL_READY_TRAIN_DIR,
    MODEL_READY_VALID_DIR,
    PIPELINE_RUN_RECORD_PATH,
)
from workflow_actions.dataset_preprocessor.source import (
    Chunker,
    RawAugment,
    SpectrogramExtractor,
    SpectrogramAugment,
    load_single_song_to_numpy,
    LabelEncoder,
    OfflineNormalizer,
    SamplePacker,
)
from workflow_actions.json_handlers import write_dict_to_json, read_json_to_dict
from typing import TYPE_CHECKING
import torch
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import shutil
from datetime import datetime

if TYPE_CHECKING:
    from workflow_actions.dataset_preprocessor.source.chunker import (
        FragmentedSongSlices,
        FragmentedSongIndex,
    )
    from numpy.typing import NDArray


@dataclass
class GlobalFragmentsIndex:
    """
    Maps song title to global (inside a given set) fragment (sample) numbers.

    For a given song it keeps first global fragment number and the last one (both inclusive).

    Songs are sorted before being added to index.
    """

    train_index: dict[str, list[int, int]]
    valid_index: dict[str, list[int, int]]

    # These keep track of number of samples already present in the given set
    _start_index_train: int = 0
    _start_index_valid: int = 0

    def add_song_to_train(self, song_title: str, n_fragments: int):
        self.train_index[song_title] = [
            self._start_index_train,
            self._start_index_train + n_fragments - 1,
        ]
        self._start_index_train += n_fragments

    def add_song_to_valid(self, song_title: str, n_fragments: int):
        self.valid_index[song_title] = [
            self._start_index_valid,
            self._start_index_valid + n_fragments - 1,
        ]
        self._start_index_valid += n_fragments

    def dump_indexes(self):
        write_dict_to_json(self.train_index, GLOBAL_TRAIN_INDEX_PATH)
        write_dict_to_json(self.valid_index, GLOBAL_VALID_INDEX_PATH)


class DatasetPreprocessor:
    def __init__(
        self,
        chunker: dict,
        raw_augment: dict,
        spectrogram_extractor: dict,
        spectrogram_augment: dict,
        sample_packer: dict,
        offline_normalizer: dict,
        pipeline_settings: dict,
    ):
        self.chunker_cfg = chunker
        self.chunker = Chunker(**self.chunker_cfg)

        self.label_encoder = LabelEncoder()

        self.raw_augment = RawAugment(**raw_augment)

        self._spectrogram_extractor_cfg = spectrogram_extractor
        self.spectrogram_extractor = SpectrogramExtractor(
            **self._spectrogram_extractor_cfg
        )

        self.spectrogram_augment = SpectrogramAugment(**spectrogram_augment)

        self.sample_packer = SamplePacker(**sample_packer)

        self.offline_normalizer_cfg = offline_normalizer
        self.offline_normalizer = OfflineNormalizer(**offline_normalizer)

        self._fragmentation_index: dict[str, "FragmentedSongIndex"] | None = None
        self._global_train_index: dict[str, list[int, int]] | None = None
        self._global_valid_index: dict[str, list[int, int]] | None = None
        # "train"/"valid" -> (MODEL_READY_TRAIN/VALID_DIR, global_train/valid_index)
        self._set_type_to_model_ready_and_global_index_map = None
        self._INDEX_PRESENT_SONGS = None

        self._broken_songs = []

        # always preserve alphabetical order
        self._SONG_ITERATION_ORDER = sorted(
            list(DOWNLOAD_DIR.iterdir()), key=lambda path: path.name
        )

        self._pipeline_settings = pipeline_settings
        self._PIPELINE_NODES_IN_ORDER = self._gather_pipeline_nodes_in_order(
            **self._pipeline_settings
        )

        if self.chunker.sample_rate != self.spectrogram_extractor.sample_rate:
            raise ValueError(
                (
                    "Chunker sample rate must match SpectrogramExtractor sample rate. ",
                    "Check dataset_preprocessor_config.json values.",
                )
            )
        if self.offline_normalizer.n_mels != self.spectrogram_extractor.n_mels:
            raise ValueError(
                (
                    "OfflineNormalizer n_mels must match SpectrogramExtractor n_mels. ",
                    "Check dataset_preprocessor_config.json values.",
                )
            )

    def create_fragmentation_for_all_songs(self):
        global_fragments_index = GlobalFragmentsIndex(
            train_index={},
            valid_index={},
        )
        fragmentation_index: dict[str, "FragmentedSongIndex"] = {}
        for song in self._SONG_ITERATION_ORDER:
            if song.is_file():
                self._create_fragmentation_for_single_song(
                    song_title=song.name,
                    global_fragments_index=global_fragments_index,
                    fragmentation_index=fragmentation_index,
                )
        self._record_fragmentation_run(
            global_fragments_index=global_fragments_index,
            fragmentation_index=fragmentation_index,
        )

    def run_pipeline(self):
        """Populates 03_model_ready with samples."""
        self._load_indexes()
        self._disc_and_fragmentation_index_integrity_check()
        self._empty_folder(MODEL_READY_VALID_DIR)
        self._empty_folder(MODEL_READY_TRAIN_DIR)
        # TODO: Implement outer loop parallely (?)
        for song in self._SONG_ITERATION_ORDER:
            if song.stem not in self._INDEX_PRESENT_SONGS:
                continue
            state = None
            for node_name, node in self._PIPELINE_NODES_IN_ORDER:
                if node_name == "_get_raw_fragments":
                    state = node(song_title=song.stem)
                elif node_name == "_serialize_samples":
                    node(state, song_title=song.stem)
                else:
                    state = node(state)
        self.offline_normalizer()
        self._record_pipeline_run()

    def pre_epoch_augment_hook(self):
        raise NotImplementedError("Implement this method.")

    def _create_fragmentation_for_single_song(
        self,
        song_title: str,
        global_fragments_index: GlobalFragmentsIndex,
        fragmentation_index: dict[str, "FragmentedSongIndex"],
    ):
        song, sample_rate = load_single_song_to_numpy(path=DOWNLOAD_DIR / song_title)
        if not sample_rate:
            self._broken_songs.append(song_title)
            return None

        song_fragmentation_index: "FragmentedSongIndex" = self.chunker.make_song_index(
            song=song
        )
        song_title = Path(song_title).stem
        fragmentation_index[song_title] = song_fragmentation_index

        global_fragments_index.add_song_to_train(
            song_title=song_title, n_fragments=len(song_fragmentation_index["train"])
        )
        global_fragments_index.add_song_to_valid(
            song_title=song_title, n_fragments=len(song_fragmentation_index["valid"])
        )

    def _get_raw_fragments(
        self,
        song_title: str,
    ) -> "FragmentedSongSlices":
        song, sample_rate = load_single_song_to_numpy(path=DOWNLOAD_DIR / song_title)
        if sample_rate != self.chunker.sample_rate:
            raise ValueError(
                "DatasetPreprocessor._get_raw_fragments:\n"
                f"Loaded song sample rate {sample_rate} doesn't match chunker sample rate {self.chunker.sample_rate}."
                "Check consistency."
            )
        raw_fragments = self.chunker.make_song_slices(
            song=song,
            fragmented_song_index=self._fragmentation_index[song_title],
        )
        return raw_fragments

    def _get_augmented_raw_fragments(
        self,
        raw_fragments: "FragmentedSongSlices",
    ) -> dict[str, list["NDArray[np.float32]"]]:
        augmented_raw_fragments = {
            "train": self.raw_augment(
                raw_fragments=raw_fragments["train"],
                sample_rate=self.chunker.sample_rate,
            ),
            "valid": self.raw_augment(
                raw_fragments=raw_fragments["valid"],
                sample_rate=self.chunker.sample_rate,
            ),
        }
        return augmented_raw_fragments

    def _get_fragments_spectrograms(
        self,
        raw_fragments: dict[str, list["NDArray[np.float32]"]],
    ) -> dict[str, list["NDArray[np.float32]"]]:
        fragments_spectrograms = {
            "train": self.spectrogram_extractor(raw_fragments["train"]),
            "valid": self.spectrogram_extractor(raw_fragments["valid"]),
        }
        return fragments_spectrograms

    @staticmethod
    def _get_samples_as_tensors(
        samples: dict[str, list["NDArray[np.float32]"]],
    ) -> dict[str, list[torch.Tensor]]:
        spectrograms_as_tensors = {
            "train": [
                torch.from_numpy(spectrogram.astype(np.float32)).unsqueeze(0)
                for spectrogram in samples["train"]
            ],
            "valid": [
                torch.from_numpy(spectrogram.astype(np.float32)).unsqueeze(0)
                for spectrogram in samples["valid"]
            ],
        }
        return spectrograms_as_tensors

    def _get_augmented_spectrograms(
        self,
        spectrograms_as_tensors: dict[str, list[torch.Tensor]],
    ) -> dict[str, list[torch.Tensor]]:
        augmented_spectrograms = {
            "train": self.spectrogram_augment(spectrograms_as_tensors["train"]),
            "valid": self.spectrogram_augment(spectrograms_as_tensors["valid"]),
        }
        return augmented_spectrograms

    def _serialize_samples(
        self,
        samples: dict[str, list[torch.Tensor]],
        song_title,
    ):
        for set_type in ["train", "valid"]:
            set_path, global_index = self._set_type_to_model_ready_and_global_index_map[
                set_type
            ]
            index_range = global_index[song_title]

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

    def _create_all_ys(self, *args, **kwargs):
        for (
            set_path,
            global_index,
        ) in self._set_type_to_model_ready_and_global_index_map.values():
            for song_title, index_range in global_index.items():
                encoded_song_tags = (
                    self.label_encoder.encode_song_labels_to_multi_hot_vector(
                        song_title=song_title,
                    )
                )
                for absolute_idx in range(index_range[0], index_range[1] + 1):
                    torch.save(
                        encoded_song_tags,
                        set_path / f"y_{absolute_idx}.pt",
                    )

    def _gather_pipeline_nodes_in_order(
        self,
        apply_augmentations_on_raw: bool,
        extract_spectrograms: bool,
        apply_augmentations_on_spectrograms: bool,
    ):
        return [
            ("_get_raw_fragments", self._get_raw_fragments),
            (
                "_get_augmented_raw_fragments",
                (
                    self._get_augmented_raw_fragments
                    if apply_augmentations_on_raw
                    else self._pass_func
                ),
            ),
            (
                "_get_fragments_spectrograms",
                (
                    self._get_fragments_spectrograms
                    if extract_spectrograms
                    else self._pass_func
                ),
            ),
            ("_get_samples_as_tensors", self._get_samples_as_tensors),
            (
                "_get_augmented_spectrograms",
                (
                    self._get_augmented_spectrograms
                    if extract_spectrograms and apply_augmentations_on_spectrograms
                    else self._pass_func
                ),
            ),
            ("_serialize_samples", self._serialize_samples),
            ("_create_all_ys", self._create_all_ys),
            ("sample_packer.pack", self.sample_packer.pack),
        ]

    def _load_indexes(self):
        try:
            self._fragmentation_index = read_json_to_dict(FRAGMENTATION_INDEX_PATH)[
                "fragmentation_index"
            ]
            self._INDEX_PRESENT_SONGS = set(list(self._fragmentation_index.keys()))
            self._global_train_index = read_json_to_dict(GLOBAL_TRAIN_INDEX_PATH)
            self._global_valid_index = read_json_to_dict(GLOBAL_VALID_INDEX_PATH)
            self._set_type_to_model_ready_and_global_index_map = {
                "train": (MODEL_READY_TRAIN_DIR, self._global_train_index),
                "valid": (MODEL_READY_VALID_DIR, self._global_valid_index),
            }
        except FileNotFoundError as e:
            print(
                f"{str(e)}.\nRun DatasetPreprocessor.create_fragmentation_for_all_songs first."
            )
            raise

    def _record_fragmentation_run(
        self,
        global_fragments_index: GlobalFragmentsIndex,
        fragmentation_index: dict[str, "FragmentedSongIndex"],
    ):
        now = datetime.now()
        formatted_time = now.strftime("%Y/%m/%d_%H-%M-%S")
        scrape_stamp = read_json_to_dict(SCRAPE_STAMP_PATH)
        fragmentation_stamp = {
            **self.chunker_cfg,
            "broken_songs": self._broken_songs,
            "scrape_stamp": scrape_stamp,
            "time_stamp": formatted_time,
        }
        write_dict_to_json(fragmentation_stamp, FRAGMENTATION_STAMP_PATH)
        write_dict_to_json(
            {"time_stamp": formatted_time, "fragmentation_index": fragmentation_index},
            FRAGMENTATION_INDEX_PATH,
        )
        global_fragments_index.dump_indexes()

    def _record_pipeline_run(self):
        cfg = self._pipeline_settings
        cfg = (
            cfg | {"spectrogram_extractor": self._spectrogram_extractor_cfg}
            if self._pipeline_settings["extract_spectrograms"]
            else cfg
        )
        cfg = cfg | {"offline_normalization": self.offline_normalizer_cfg}
        write_dict_to_json(cfg, PIPELINE_RUN_RECORD_PATH)

    def _disc_and_fragmentation_index_integrity_check(self):
        if not self._INDEX_PRESENT_SONGS:
            raise RuntimeError(
                "before running _disc_and_fragmentation_index_integrity_check load_indexes must be run."
            )

        disk_present_songs = set(
            read_json_to_dict(SCRAPE_STAMP_PATH)["downloaded_songs"]
        )
        missing = self._INDEX_PRESENT_SONGS - disk_present_songs
        if missing:
            raise RuntimeError(
                f"Some songs from fragmentation index are not present on disk: {missing}"
            )

    @staticmethod
    def _pass_func(x):
        return x

    @staticmethod
    def _empty_folder(path: Path):
        """Removes everything in `path` folder"""
        for element in path.iterdir():
            if element.is_file():
                element.unlink()
            elif element.is_dir():
                shutil.rmtree(element)
