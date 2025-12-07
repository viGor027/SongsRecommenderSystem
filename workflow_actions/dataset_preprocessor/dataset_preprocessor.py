from workflow_actions.paths import (
    DOWNLOAD_DIR,
    FRAGMENTATION_STAMP_PATH,
    FRAGMENTATION_INDEX_PATH,
    SCRAPE_STAMP_PATH,
    MODEL_READY_TRAIN_DIR,
    MODEL_READY_VALID_DIR,
    PIPELINE_RUN_RECORD_PATH,
)
from workflow_actions.dataset_preprocessor.source import (
    Chunker,
    FragmentPipeline,
    Serializer,
    GlobalFragmentsIndex,
    SamplePacker,
)
from workflow_actions.json_handlers import write_dict_to_json, read_json_to_dict
from typing import TYPE_CHECKING
from pathlib import Path
import shutil
from datetime import datetime

if TYPE_CHECKING:
    from workflow_actions.dataset_preprocessor.source.chunker import (
        FragmentedSongSlices,
        FragmentedSongIndex,
    )


class DatasetPreprocessor:
    def __init__(
        self,
        chunker: dict,
        fragment_pipeline: dict,
        sample_packer: dict,
    ):
        self.chunker_cfg = chunker
        self.fragment_pipeline_cfg = fragment_pipeline

        self.chunker = Chunker(**self.chunker_cfg)
        self.fragment_pipeline = FragmentPipeline(**self.fragment_pipeline_cfg)

        if (
            self.chunker.sample_rate
            != self.fragment_pipeline.spectrogram_extractor.sample_rate
        ):
            raise ValueError(
                (
                    "Chunker sample rate must match SpectrogramExtractor sample rate. ",
                    "Check dataset_preprocessor_config.json values.",
                )
            )
        else:
            _sample_rate = self.fragment_pipeline.spectrogram_extractor.sample_rate

        self.serializer = Serializer(load_sample_rate=_sample_rate)
        self._ys_already_created = False

        self.sample_packer = SamplePacker(**sample_packer)

        self._fragmentation_index: dict[str, "FragmentedSongIndex"] | None = (
            self._load_fragmentation_index()
        )
        self._index_present_songs = set(list(self._fragmentation_index.keys()))

        self._broken_songs = []

        # always preserve alphabetical order
        self._SONG_ITERATION_ORDER = sorted(
            list(DOWNLOAD_DIR.iterdir()), key=lambda path: path.name
        )

    def create_fragmentation_for_all_songs(self):
        global_fragments_index = GlobalFragmentsIndex()
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
        self._assure_local_song_files_match_fragmentation_index()
        self._empty_folder(MODEL_READY_VALID_DIR)
        self._empty_folder(MODEL_READY_TRAIN_DIR)
        # TODO: Implement outer loop parallely (?)
        for song in self._SONG_ITERATION_ORDER:
            if song.stem not in self._index_present_songs:
                continue
            self._run_pipeline_for_single_song(song=song, augment=False)
        self.serializer.create_all_ys()
        self.sample_packer.pack()
        write_dict_to_json(self.fragment_pipeline_cfg, PIPELINE_RUN_RECORD_PATH)

    def pre_epoch_augment_hook(self):
        self._empty_folder(MODEL_READY_VALID_DIR)
        self._empty_folder(MODEL_READY_TRAIN_DIR)
        for song in self._SONG_ITERATION_ORDER:
            if song.stem not in self._index_present_songs:
                continue
            self._run_pipeline_for_single_song(song=song, augment=True)
        if not self._ys_already_created:
            self.serializer.create_all_ys()
            self._ys_already_created = True

    def _create_fragmentation_for_single_song(
        self,
        song_title: str,
        global_fragments_index: GlobalFragmentsIndex,
        fragmentation_index: dict[str, "FragmentedSongIndex"],
    ):
        song, sample_rate = self.serializer.load_single_song_to_numpy(
            path=DOWNLOAD_DIR / song_title
        )
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

    def _run_pipeline_for_single_song(self, song: Path, augment: bool):
        """Pass song with extension."""
        song_fragments = self._load_fragments_for_song(song_title=song.stem)
        train_fgs, valid_fgs = song_fragments["train"], song_fragments["valid"]
        train_specs = self.fragment_pipeline.process_raw_fragments(
            fragments=train_fgs,
            augment=augment,
        )
        valid_specs = self.fragment_pipeline.process_raw_fragments(
            fragments=valid_fgs,
            augment=False,
        )
        self.serializer.serialize_song_samples(
            song_title=song.stem,
            samples={
                "train": train_specs,
                "valid": valid_specs,
            },
        )

    def _load_fragments_for_song(
        self,
        song_title: str | Path,
    ) -> "FragmentedSongSlices":
        song, sample_rate = self.serializer.load_single_song_to_numpy(
            path=DOWNLOAD_DIR / song_title
        )
        fragments = self.chunker.make_song_slices(
            song=song,
            fragmented_song_index=self._fragmentation_index[song_title],
        )
        return fragments

    @staticmethod
    def _load_fragmentation_index():
        try:
            return read_json_to_dict(FRAGMENTATION_INDEX_PATH)["fragmentation_index"]
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

    def _assure_local_song_files_match_fragmentation_index(self):
        if not self._index_present_songs:
            raise RuntimeError(
                "before running _disc_and_fragmentation_index_integrity_check load_indexes must be run."
            )

        disk_present_songs = set(
            read_json_to_dict(SCRAPE_STAMP_PATH)["downloaded_songs"]
        )
        missing = self._index_present_songs - disk_present_songs
        if missing:
            raise RuntimeError(
                f"Some songs from fragmentation index are not present on disk: {missing}"
            )

    @staticmethod
    def _empty_folder(path: Path):
        """Removes everything in `path` folder"""
        for element in path.iterdir():
            if element.is_file():
                element.unlink()
            elif element.is_dir():
                shutil.rmtree(element)
