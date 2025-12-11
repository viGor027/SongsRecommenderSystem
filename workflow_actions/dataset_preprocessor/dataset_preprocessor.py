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
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

if TYPE_CHECKING:
    from workflow_actions.dataset_preprocessor.source.chunker import (
        FragmentedSongSlices,
        FragmentedSongIndex,
    )


def _process_single_song_mp(
    song_path_str: str, chunker_cfg: dict, pipeline_cfg: dict, affect_valid: bool
):
    from pathlib import Path
    from workflow_actions.paths import DOWNLOAD_DIR, FRAGMENTATION_INDEX_PATH
    from workflow_actions.json_handlers import read_json_to_dict
    from workflow_actions.dataset_preprocessor.source import (
        Chunker,
        FragmentPipeline,
        Serializer,
    )

    song_path = Path(song_path_str)
    song_title = song_path.stem

    chunker = Chunker(**chunker_cfg)
    fragment_pipeline = FragmentPipeline(**pipeline_cfg)
    sample_rate = fragment_pipeline.spectrogram_extractor.sample_rate
    serializer = Serializer(load_sample_rate=sample_rate)

    frag_index = read_json_to_dict(FRAGMENTATION_INDEX_PATH)["fragmentation_index"][
        song_title
    ]

    song, sr = serializer.load_single_song_to_numpy(DOWNLOAD_DIR / song_path.name)

    fragments = chunker.make_song_slices(
        song=song,
        fragmented_song_index=frag_index,
        get_valid_slices=affect_valid,
    )
    train_fgs = fragments["train"]
    valid_fgs = fragments["valid"]

    train_specs = fragment_pipeline.process_raw_fragments(train_fgs, augment=True)
    valid_specs = (
        fragment_pipeline.process_raw_fragments(valid_fgs, augment=False)
        if affect_valid
        else []
    )

    serializer.serialize_song_samples(
        song_title=song_title,
        samples={"train": train_specs, "valid": valid_specs},
        serialize_valid=affect_valid,
    )


def _process_single_song_mp_wrapper(args: tuple[str, dict, dict, bool]):
    song_path_str, chunker_cfg, pipeline_cfg, affect_valid = args
    _process_single_song_mp(song_path_str, chunker_cfg, pipeline_cfg, affect_valid)


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

        self.sample_packer = SamplePacker(**sample_packer)

        self._fragmentation_index: dict[str, "FragmentedSongIndex"] | None = (
            self._load_fragmentation_index()
        )
        self._index_present_songs = set(list(self._fragmentation_index.keys()))

        self._broken_songs = []

        # always preserve alphabetical order
        self._SONGS_ITERATION_ORDER = sorted(
            list(DOWNLOAD_DIR.iterdir()), key=lambda path: path.name
        )

        self._first_pre_epoch_hook_run = True

    def create_fragmentation_for_all_songs(self):
        global_fragments_index = GlobalFragmentsIndex()
        fragmentation_index: dict[str, "FragmentedSongIndex"] = {}
        for song in self._SONGS_ITERATION_ORDER:
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

    def run_pipeline(self, for_sanity_check: bool = False, augment: bool = False):
        """Populates 03_model_ready with samples."""
        self._assure_local_song_files_match_fragmentation_index()
        self._remove_samples_from_dir(MODEL_READY_VALID_DIR, delete_ys=True)
        self._remove_samples_from_dir(MODEL_READY_TRAIN_DIR, delete_ys=True)
        songs_to_process = (
            self._SONGS_ITERATION_ORDER
            if not for_sanity_check
            else self._SONGS_ITERATION_ORDER[:1]
        )
        for song in songs_to_process:
            if song.stem not in self._index_present_songs:
                continue
            self._run_pipeline_for_single_song(song=song, augment=augment)

        if not for_sanity_check:
            self.serializer.create_all_ys()
        else:
            self.serializer.create_single_song_ys(song_title=songs_to_process[0].stem)

        self.sample_packer.pack()
        write_dict_to_json(self.fragment_pipeline_cfg, PIPELINE_RUN_RECORD_PATH)

    def pre_epoch_augment_hook(self, max_workers=8):
        if self._first_pre_epoch_hook_run:
            self._remove_samples_from_dir(MODEL_READY_VALID_DIR, delete_ys=True)
            self._remove_samples_from_dir(MODEL_READY_TRAIN_DIR, delete_ys=True)
        else:
            self._remove_samples_from_dir(MODEL_READY_TRAIN_DIR, delete_ys=False)

        songs_to_process = [
            song
            for song in self._SONGS_ITERATION_ORDER
            if song.stem in self._index_present_songs
        ]
        affect_valid = self._first_pre_epoch_hook_run
        jobs = [
            (str(song), self.chunker_cfg, self.fragment_pipeline_cfg, affect_valid)
            for song in songs_to_process
        ]

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            list(ex.map(_process_single_song_mp_wrapper, jobs))

        if self._first_pre_epoch_hook_run:
            self.serializer.create_all_ys()
            self._first_pre_epoch_hook_run = False

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
            serialize_valid=True,
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
            get_valid_slices=True,
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
    def _remove_samples_from_dir(path: Path, delete_ys: bool):
        for element in path.glob("X_*.pt"):
            if element.is_file():
                element.unlink()

        if delete_ys:
            for element in path.glob("y_*.pt"):
                if element.is_file():
                    element.unlink()
