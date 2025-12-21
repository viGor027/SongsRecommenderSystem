from workflow_actions.paths import (
    DOWNLOAD_DIR,
    FRAGMENTATION_STAMP_PATH,
    FRAGMENTATION_INDEX_PATH,
    FRAGMENTED_DATA_DIR,
    SCRAPE_STAMP_PATH,
    MODEL_READY_TRAIN_DIR,
    MODEL_READY_VALID_DIR,
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
    song_str_with_extension: str,
    chunker_cfg: dict,
    pipeline_cfg: dict,
    serializer_cfg: dict,
    affect_valid: bool,
):
    from pathlib import Path
    from workflow_actions.paths import DOWNLOAD_DIR, FRAGMENTATION_INDEX_PATH
    from workflow_actions.json_handlers import read_json_to_dict
    from workflow_actions.dataset_preprocessor.source import (
        Chunker,
        FragmentPipeline,
        Serializer,
    )

    song_path_obj = Path(song_str_with_extension)
    song_title = song_path_obj.stem

    chunker = Chunker(**chunker_cfg)
    fragment_pipeline = FragmentPipeline(**pipeline_cfg)
    serializer = Serializer(**serializer_cfg)

    frag_index = read_json_to_dict(FRAGMENTATION_INDEX_PATH)["fragmentation_index"][
        song_title
    ]

    song, sr = serializer.load_single_song_to_numpy(DOWNLOAD_DIR / song_path_obj.name)

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


def _process_single_song_mp_wrapper(args: tuple[str, dict, dict, dict, bool]):
    song_str_with_extension, chunker_cfg, pipeline_cfg, serializer_cfg, affect_valid = (
        args
    )
    _process_single_song_mp(
        song_str_with_extension, chunker_cfg, pipeline_cfg, serializer_cfg, affect_valid
    )


class DatasetPreprocessor:
    def __init__(
        self,
        n_workers: int,
        chunker: dict,
        fragment_pipeline: dict,
        sample_packer: dict,
        serializer: dict,
    ):
        self.n_workers = n_workers

        self.chunker_cfg = chunker
        self.fragment_pipeline_cfg = fragment_pipeline

        self.chunker = Chunker(**self.chunker_cfg)
        self.fragment_pipeline = FragmentPipeline(**self.fragment_pipeline_cfg)

        self.serializer_cfg = serializer
        self.serializer = Serializer(**self.serializer_cfg)

        self.sample_packer = SamplePacker(**sample_packer)

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

        self._packed_fragmentation_index = self._load_fragmentation_index()
        self._fragmentation_index: dict[str, "FragmentedSongIndex"] | None = (
            self._packed_fragmentation_index[0]
        )
        self._fragmentation_index_time_stamp = self._packed_fragmentation_index[1]
        self._index_present_songs = set(list(self._fragmentation_index.keys()))

        self._broken_songs = []

        # always preserve alphabetical order
        self._SONGS_ITERATION_ORDER = sorted(
            list(DOWNLOAD_DIR.iterdir()), key=lambda path: path.name
        )

        self._first_pipeline_run = True

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

    def prepare_model_ready_data(self, for_sanity_check: bool = False):
        """
        Notes:
         - This method has two intended usages: offline full build (first run) and online refresh between epochs (subsequent runs).
         - Online usage assumes SamplePacker.group_size is None (pack is a no-op); only train X_* are regenerated and y_*/valid are reused.
         - Offline usage (first run) removes train+valid X_*/y_*, regenerates data and y_*; optional packing is done only when SamplePacker.group_size is an int.
         - for_sanity_check is offline-only.
        """
        if self._first_pipeline_run:
            self._assure_local_song_files_match_fragmentation_index()
            self._remove_samples_from_dir(MODEL_READY_VALID_DIR, delete_ys=True)
            self._remove_samples_from_dir(MODEL_READY_TRAIN_DIR, delete_ys=True)
        else:
            self._remove_samples_from_dir(MODEL_READY_TRAIN_DIR, delete_ys=False)

        index_and_disk_present_songs = [
            song
            for song in self._SONGS_ITERATION_ORDER
            if song.stem in self._index_present_songs
        ]

        songs_to_process = (
            index_and_disk_present_songs[:1]
            if for_sanity_check
            else index_and_disk_present_songs
        )

        affect_valid = self._first_pipeline_run
        jobs = [
            (
                str(song),
                self.chunker_cfg,
                self.fragment_pipeline_cfg,
                self.serializer_cfg,
                affect_valid,
            )
            for song in songs_to_process
        ]

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx) as ex:
            list(ex.map(_process_single_song_mp_wrapper, jobs))

        if self._first_pipeline_run:
            if for_sanity_check:
                self.serializer.create_single_song_ys(
                    song_title=songs_to_process[0].stem,
                )
                self._first_pipeline_run = True
            else:
                self.serializer.create_all_ys()
                self._first_pipeline_run = False

        self.sample_packer.pack()

    @staticmethod
    def create_non_overlapping_index_from_overlapping(
        overlapping_index: dict[str, list[int, int]],
        overlapping_index_time_stamp: str,
    ):
        now = datetime.now()
        new_index = {
            "time_stamp": now.strftime("%Y/%m/%d_%H-%M-%S"),
            "from_index": overlapping_index_time_stamp,
            "fragmentation_index": {},
        }
        for song_idx, (song_title, slices) in enumerate(overlapping_index.items()):
            new_song_idx = {
                "train": [],
                "valid": [],
            }
            for set_type in ["train", "valid"]:
                current_slices = slices[set_type]
                new_slices = []
                current_slice = current_slices[0]
                new_slices.append(current_slice)
                for slice_idx in range(1, len(current_slices)):
                    previous_slice_end = new_slices[-1][1]
                    current_slice_beginning = current_slices[slice_idx][0]
                    if current_slice_beginning >= previous_slice_end:
                        new_slices.append(current_slices[slice_idx])
                new_song_idx[set_type] = new_slices
            new_index["fragmentation_index"][song_title] = new_song_idx
        write_dict_to_json(
            new_index, FRAGMENTED_DATA_DIR / "non_overlapping_fragmentation_index.json"
        )

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
            full_idx = read_json_to_dict(FRAGMENTATION_INDEX_PATH)
            return full_idx["fragmentation_index"], full_idx["time_stamp"]
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
