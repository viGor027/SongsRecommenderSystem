from workflow_actions.dataset_preprocessor.source import LabelEncoder
from workflow_actions.paths import FRAGMENTATION_INDEX_PATH
from workflow_actions.json_handlers import read_json_to_dict
from .metric import Metric
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from architectures.assemblies.assembly import Assembly


class AccuraccyTestFullSongs(Metric):
    def __init__(
        self,
        model: "Assembly",
        model_name: str,
        index: dict[str, list[int, int]],
        Xs: list[torch.Tensor],
    ):
        super().__init__(
            model=model,
            model_name=model_name,
            index=index,
            Xs=Xs,
        )
        self.le = LabelEncoder()
        self._fragmentation_index = read_json_to_dict(FRAGMENTATION_INDEX_PATH)[
            "fragmentation_index"
        ]
        self.songs = list(index.keys())
        self.n_songs = len(self.songs)

    def __call__(self, **kwargs):
        total_labels = 0
        correctly_predicted_labels = 0

        n_exact_matches = 0
        one2nine_labels_wrong = 0
        ten2ninety_nine_labels_wrong = 0
        over_hundred_labels_wrong = 0
        for song in self.songs:
            print(f"model: {self.model_name}, song: {song}")
            song_labels = self.le.encode_song_labels_to_multi_hot_vector(
                song_title=song
            ).reshape(-1)

            idx = list(self._fragmentation_index.keys()).index(song)
            predicted_ppbs = self.model.forward(self.Xs[idx])
            predicted_labels = (predicted_ppbs > 0.5).float().reshape(-1)

            correctly_predicted_labels += int(
                (song_labels == predicted_labels).float().sum().item()
            )
            total_labels += len(song_labels)

            mismatches = int((song_labels != predicted_labels).float().sum().item())

            if mismatches == 0:
                n_exact_matches += 1
            elif mismatches <= 9:
                one2nine_labels_wrong += 1
            elif mismatches < 100:
                ten2ninety_nine_labels_wrong += 1
            else:
                over_hundred_labels_wrong += 1

        return {
            "micro_accuracy": correctly_predicted_labels / total_labels,
            "exact_matches": n_exact_matches / self.n_songs,
            "1-9 labels wrong": one2nine_labels_wrong / self.n_songs,
            "10-99 labels wrong": ten2ninety_nine_labels_wrong / self.n_songs,
            "100+ labels wrong": over_hundred_labels_wrong / self.n_songs,
        }

    def __repr__(self) -> str:
        return f"AccuraccyTestFullSongs for model {self.model_name}"
