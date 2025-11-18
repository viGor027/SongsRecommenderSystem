from abc import ABC, abstractmethod
from typing import Literal
from architectures.model_components.classifier.base_classifier import BaseClassifier


class Assembly(ABC):
    def __init__(self):
        """All the below attributes are set during initialization of child assemblies."""
        self.n_classifier_layers = None
        self.n_units_per_classifier_layer = None
        self.classifier_activation = None
        self.sigmoid_output = None
        self.n_classes = None

        self.classifier = None

    @abstractmethod
    def _classifier_in_features(self) -> int:
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_instance_config(self) -> dict:
        pass

    @abstractmethod
    def _get_normalization_layer(self):
        pass

    def init_classifier(
        self,
        n_classifier_layers: int,
        n_units_per_classifier_layer: list[int],
        n_classes: int,
        classifier_activation: Literal["relu", "hardswish"] = None,
        sigmoid_output: bool = True,
    ):
        self.n_classifier_layers = n_classifier_layers
        self.n_units_per_classifier_layer = n_units_per_classifier_layer
        self.n_classes = n_classes
        self.classifier_activation = classifier_activation
        self.sigmoid_output = sigmoid_output
        self.classifier = self._build_classifier()

    def _build_classifier(self):
        classifier = BaseClassifier(
            n_layers=self.n_classifier_layers,
            n_input_features=self._classifier_in_features(),
            units_per_layer=self.n_units_per_classifier_layer,
            activation=self.classifier_activation,
            n_classes=self.n_classes,
            sigmoid_output=self.sigmoid_output,
        )
        return classifier

    def get_classifier_config(self) -> dict:
        return {
            "n_classifier_layers": self.n_classifier_layers,
            "n_units_per_classifier_layer": self.n_units_per_classifier_layer,
            "n_classes": self.n_classes,
            "classifier_activation": self.classifier_activation,
            "sigmoid_output": self.sigmoid_output,
        }
