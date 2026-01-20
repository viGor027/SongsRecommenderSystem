from abc import ABC, abstractmethod
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from architectures.assemblies.assembly import Assembly


class Metric(ABC):
    def __init__(
        self,
        model: "Assembly",
        model_name: str,
        index: dict[str, list[int, int]],
        Xs: list[torch.Tensor],
    ):
        """
        Index maps song title to global fragment numbers inside space.

        Note: MODEL_READY_TRAIN_DIR has only train samples for embedding space `space` as validation probability is 0 for this space fragmentation
        """
        self.model = model
        self.model.eval()
        self.model_name = model_name
        self.index = index
        self.Xs = Xs

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass
