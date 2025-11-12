import torch.nn as nn
from architectures.model_components.classifier.base_classifier import BaseClassifier
from architectures.assemblies.assembly import Assembly
from typing import Literal


class DenseAssembly(nn.Module, Assembly):
    """
    A wrapper for convenient model assembling.

    You must initialize each part of the model before usage:
    '''
        model = DenseAssembly()
        model.init_feature_extractor(...)
        model.init_classifier(...)
    '''
    """

    def __init__(self):
        """All the below attributes are set during initialization mentioned in class docstring."""
        nn.Module.__init__(self)
        Assembly.__init__(self)

        self.n_input_channels = None
        self.n_feature_extractor_layers = None
        self.n_units_per_feature_extractor_layer = None
        self.feature_extractor_activation = None
        self.n_embedding_dims = None

        self.feature_extractor = None

        self.input_normalization_layer = None

    def init_feature_extractor(
        self,
        n_input_channels: int,
        n_feature_extractor_layers: int,
        n_units_per_feature_extractor_layer: list[int],
        n_embedding_dims: int,
        feature_extractor_activation: Literal["relu", "hardswish"] = "relu",
    ):
        self.n_input_channels = n_input_channels
        self.n_feature_extractor_layers = n_feature_extractor_layers
        self.n_units_per_feature_extractor_layer = n_units_per_feature_extractor_layer
        self.feature_extractor_activation = feature_extractor_activation
        self.n_embedding_dims = n_embedding_dims
        self.feature_extractor = self._build_feature_extractor()

        self.input_normalization_layer = self._get_normalization_layer()

    def _build_feature_extractor(self):
        return BaseClassifier(
            n_layers=self.n_feature_extractor_layers,
            n_input_features=self.n_input_channels,
            units_per_layer=self.n_units_per_feature_extractor_layer,
            activation=self.feature_extractor_activation,
            n_classes=self.n_embedding_dims,
            sigmoid_output=False,
        )

    def _classifier_in_features(self) -> int:
        return self.n_embedding_dims

    def _get_normalization_layer(self):
        return nn.BatchNorm1d(num_features=self.n_input_channels)

    def forward(self, x):
        x = self.input_normalization_layer(x)
        x = x.reshape((x.size(0), -1))
        x = self.feature_extractor(x)
        out = self.classifier(x)
        return out

    def get_instance_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "feature_extractor": {
                "n_input_channels": self.n_input_channels,
                "n_feature_extractor_layers": self.n_feature_extractor_layers,
                "n_units_per_feature_extractor_layer": self.n_units_per_feature_extractor_layer,
                "feature_extractor_activation": self.feature_extractor_activation,
                "n_embedding_dims": self.n_embedding_dims,
            },
            "classifier": self.get_classifier_config(),
        }
