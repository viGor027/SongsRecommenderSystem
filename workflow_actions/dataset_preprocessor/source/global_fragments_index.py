from workflow_actions.paths import (
    GLOBAL_TRAIN_INDEX_PATH,
    GLOBAL_VALID_INDEX_PATH,
)
from dataclasses import dataclass, field
from workflow_actions.json_handlers import write_dict_to_json, read_json_to_dict


@dataclass
class GlobalFragmentsIndex:
    """
    Maps song title to global (inside a given set) fragment (sample) numbers.

    For a given song it keeps first global fragment number and the last one (both inclusive).

    Songs are sorted before being added to index.
    """

    train_index: dict[str, list[int, int]] = field(default_factory=lambda: {})
    valid_index: dict[str, list[int, int]] = field(default_factory=lambda: {})

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

    def load_indexes(self):
        self.train_index = read_json_to_dict(GLOBAL_TRAIN_INDEX_PATH)
        self.valid_index = read_json_to_dict(GLOBAL_VALID_INDEX_PATH)
