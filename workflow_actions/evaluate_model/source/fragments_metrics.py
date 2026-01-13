from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import torch
import random
import torch.nn.functional as F

if TYPE_CHECKING:
    from architectures.assemblies.assembly import Assembly


class FragmentsMetric(ABC):
    def __init__(
        self,
        model: "Assembly",
        model_name: str,
        space: torch.Tensor,
        index: dict[str, list[int, int]],
    ):
        self.model = model
        self.model_name = model_name
        self.space = space
        self.index = index

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class RandomizedABXTest(FragmentsMetric):
    def __init__(
        self,
        model: "Assembly",
        model_name: str,
        space: torch.Tensor,
        index: dict[str, list[int, int]],
    ):
        """index maps song title to global fragment numbers inside space."""
        super().__init__(
            model=model,
            space=space,
            model_name=model_name,
            index=index,
        )
        self.songs = list(index.keys())
        self.k_triplets = None

    def __call__(self, **kwargs):
        n_songs = len(self.songs)
        self.k_triplets = kwargs["randomized_abx_test_k_triplets"]

        triplets, labels = self._get_triplets(
            n_songs=n_songs, k_triplets=self.k_triplets
        )

        a_idx = torch.tensor([t[0] for t in triplets])
        b_idx = torch.tensor([t[1] for t in triplets])
        x_idx = torch.tensor([t[2] for t in triplets])

        A = self.space[a_idx]
        B = self.space[b_idx]
        X = self.space[x_idx]

        sim_AX = F.cosine_similarity(A, X)
        sim_BX = F.cosine_similarity(B, X)

        preds = sim_BX > sim_AX
        labels_tensor = torch.tensor(labels, dtype=torch.bool)

        acc = (preds == labels_tensor).float().mean().item()
        return acc

    def __repr__(self) -> str:
        return f"RandomizedABXTest with {self.k_triplets} for model {self.model_name}"

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
