from typing import TYPE_CHECKING
import torch
import random
import torch.nn.functional as F
from workflow_actions.dataset_preprocessor.source import LabelEncoder
from .metric import Metric

if TYPE_CHECKING:
    from architectures.assemblies.assembly import Assembly


class RandomizedABXTest(Metric):
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
        self.songs = list(index.keys())
        self.k_triplets = None

    def __call__(self, **kwargs):
        n_songs = len(self.songs)
        self.k_triplets = kwargs["randomized_abx_test_k_triplets"]

        triplets, labels = self._get_triplets(
            n_songs=n_songs, k_triplets=self.k_triplets
        )

        A, B, X = [], [], []
        for a_idx, b_idx, x_idx in triplets:
            A.append(self.model.make_embeddings(self.Xs[a_idx]))
            B.append(self.model.make_embeddings(self.Xs[b_idx]))
            X.append(self.model.make_embeddings(self.Xs[x_idx]))

        A = torch.cat(A)
        B = torch.cat(B)
        X = torch.cat(X)

        sim_AX = F.cosine_similarity(A, X)
        sim_BX = F.cosine_similarity(B, X)

        preds = sim_BX > sim_AX
        labels_tensor = torch.tensor(labels, dtype=torch.bool)

        acc = (preds == labels_tensor).float().mean().item()
        return {"abx_accuracy": acc}

    def __repr__(self) -> str:
        return f"RandomizedABXTest with {self.k_triplets} triplets for model {self.model_name}"

    def _get_triplets(self, n_songs: int, k_triplets: int):
        """
        Returns:
          - triplets: list of (a_idx, b_idx, x_idx) – fragments indexes in self.space
          - labels:   0 if X belongs to the same song as A, 1 if same as B
        """
        triplets: list[tuple[int, int, int]] = []
        labels: list[int] = []

        for _ in range(k_triplets):
            A_song_idx, B_song_idx = random.sample(range(n_songs), 2)
            song_titles = [self.songs[A_song_idx], self.songs[B_song_idx]]

            two_from = random.randrange(2)
            one_from = 1 - two_from

            two_title = song_titles[two_from]
            one_title = song_titles[one_from]

            start_two, end_two = self.index[two_title]
            start_one, end_one = self.index[one_title]

            two_fragments = random.sample(range(start_two, end_two + 1), 2)
            one_fragment = random.sample(range(start_one, end_one + 1), 1)[0]

            if two_from == 0:
                # A and X from song_titles[0], B from song_titles[1]
                a_idx = two_fragments[0]
                b_idx = one_fragment
                x_idx = two_fragments[1]
                label = 0
            else:
                # B, X from song_titles[1], A from song_titles[0]
                a_idx = one_fragment
                b_idx = two_fragments[0]
                x_idx = two_fragments[1]
                label = 1

            triplets.append((a_idx, b_idx, x_idx))
            labels.append(label)

        return triplets, labels


class AccuraccyTest(Metric):
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
        self.songs = list(index.keys())
        self.total_number_of_fragments = sum(
            (end - start + 1) for start, end in self.index.values()
        )

    def __call__(self, **kwargs):
        total_labels = 0
        correctly_predicted_labels = 0

        n_exact_matches = 0
        one2nine_labels_wrong = 0
        ten2ninety_nine_labels_wrong = 0
        over_hundred_labels_wrong = 0
        for song in self.songs:
            song_labels = self.le.encode_song_labels_to_multi_hot_vector(
                song_title=song
            ).reshape(-1)
            for idx in range(self.index[song][0], self.index[song][1] + 1):
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
            "exact_matches": n_exact_matches / self.total_number_of_fragments,
            "1-9 labels wrong": one2nine_labels_wrong / self.total_number_of_fragments,
            "10-99 labels wrong": ten2ninety_nine_labels_wrong
            / self.total_number_of_fragments,
            "100+ labels wrong": over_hundred_labels_wrong
            / self.total_number_of_fragments,
        }

    def __repr__(self) -> str:
        return f"AccuraccyTest for model {self.model_name}"
