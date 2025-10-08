from abc import ABC, abstractmethod


class Assembly(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_instance_config(self) -> dict:
        pass
