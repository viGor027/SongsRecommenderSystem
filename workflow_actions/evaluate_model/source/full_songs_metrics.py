from workflow_actions.dataset_preprocessor.source import LabelEncoder
from workflow_actions.paths import FRAGMENTATION_INDEX_PATH
from workflow_actions.json_handlers import read_json_to_dict
from .metric import Metric
import torch
from typing import TYPE_CHECKING
from tqdm import tqdm
from collections import Counter

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
        self.song2idx = {
            song: i for i, song in enumerate(list(self._fragmentation_index.keys()))
        }
        self.n_songs = len(self.songs)

    def __call__(self, **kwargs):
        total_labels = 0
        correctly_predicted_labels = 0

        mismatches_list = []
        song_prediction_probabilities_vectors = {}

        for song in tqdm(self.songs):
            song_labels = self.le.encode_song_labels_to_multi_hot_vector(
                song_title=song
            ).reshape(-1)

            predicted_ppbs = self.model.forward(self.Xs[self.song2idx[song]]).reshape(
                -1
            )
            song_prediction_probabilities_vectors[song] = predicted_ppbs.tolist()
            print(song_prediction_probabilities_vectors[song])
            predicted_labels = (predicted_ppbs > 0.5).float()

            correctly_predicted_labels += int(
                (song_labels == predicted_labels).float().sum().item()
            )
            total_labels += len(song_labels)

            mismatches_list.append(
                int((song_labels != predicted_labels).float().sum().item())
            )

        return {
            "song predictions probabilities vectors": song_prediction_probabilities_vectors,
            "mismatches": dict(Counter(mismatches_list)),
        }

    def __repr__(self) -> str:
        return f"AccuraccyTestFullSongs for model {self.model_name}"
