import torch.nn as nn
from architectures.model_components.classifier.base_classifier import BaseClassifier
from architectures.assemblies.assembly import Assembly
from typing import Literal


class RnnDenseAssembly(nn.Module, Assembly):
    """
    A wrapper for convenient model assembling.

    You must initialize each part of the model before usage:
    '''
        model = RnnDenseAssembly()
        model.init_seq_encoder(...)
        model.init_classifier(...)
    '''
    """

    def __init__(self):
        """All the below attributes are set during initialization mentioned in class docstring."""
        super().__init__()

        self.input_len = None
        self.n_input_channels = None
        self.n_seq_encoder_layers = None
        self.hidden_size = None
        self.seq_encoder_dropout = None
        self.seq_encoder_layer_type = None

        self.n_classifier_layers = None
        self.n_units_per_classifier_layer = None
        self.n_classes = None

        self.seq_encoder = None
        self.classifier = None

        self.forward_func = None

    def init_conv(self, **kwargs):
        """Method added for API consistency."""
        pass

    def init_seq_encoder(
        self,
        n_input_channels: int,
        n_seq_encoder_layers: int,
        hidden_size: int,
        dropout: float,
        layer_type: Literal["gru", "lstm"],
    ):
        """
        Args:
            n_input_channels (int): Number of input channels.
            n_seq_encoder_layers (int): Number of layers in the sequence encoder (GRU or LSTM).
            hidden_size (int): Hidden state size of the sequence encoder.
            dropout (float): Dropout probability for the sequence encoder.
            layer_type (Literal['gru', 'lstm']): Type of sequence encoder ('gru' or 'lstm').
        """
        self.n_input_channels = n_input_channels
        self.n_seq_encoder_layers = n_seq_encoder_layers
        self.hidden_size = hidden_size
        self.seq_encoder_dropout = dropout
        self.seq_encoder_layer_type = layer_type
        self.seq_encoder = self._build_seq_encoder()

        self.forward_func = (
            self._forward_gru if layer_type == "gru" else self._forward_lstm
        )

    def _build_seq_encoder(self):
        """
        Builds sequence encoder based on configuration passed to init_seq_encoder.

        Returns:
            nn.Sequential: Sequential container of recurrent layers.
        """
        if self.seq_encoder_layer_type == "gru":
            return nn.Sequential(
                nn.GRU(
                    input_size=self.n_input_channels,
                    hidden_size=self.hidden_size,
                    dropout=self.seq_encoder_dropout,
                    num_layers=self.n_seq_encoder_layers,
                    batch_first=True,
                )
            )
        else:
            return nn.Sequential(
                nn.LSTM(
                    input_size=self.n_input_channels,
                    hidden_size=self.hidden_size,
                    dropout=self.seq_encoder_dropout,
                    num_layers=self.n_seq_encoder_layers,
                    batch_first=True,
                )
            )

    def init_classifier(
        self,
        n_classifier_layers: int,
        n_units_per_classifier_layer: list[int],
        n_classes: int,
    ):
        """
        Args:
            n_classifier_layers (int): Number of layers in the classifier.
            n_units_per_classifier_layer (list[int]): List of the number of units per classifier layer.
            n_classes (int): Number of output classes for classification.
        """
        self.n_classifier_layers = n_classifier_layers
        self.n_units_per_classifier_layer = n_units_per_classifier_layer
        self.n_classes = n_classes
        self.classifier = self._build_class()

    def _build_class(self):
        """
        Builds sequence encoder based on configuration passed to init_classifier.

        Returns:
            BaseClassifier: Classifier network.
        """
        n_input_features = self.hidden_size

        classifier = BaseClassifier(
            n_layers=self.n_classifier_layers,
            n_input_features=n_input_features,
            units_per_layer=self.n_units_per_classifier_layer,
            n_classes=self.n_classes,
        )
        return classifier

    def forward(self, x):
        x = self.forward_func(x)
        return x

    def _forward_gru(self, x):
        x = x.permute(0, 2, 1)
        _, h_n = self.seq_encoder(x)
        x = h_n[-1]
        x = self.classifier(x)
        return x

    def _forward_lstm(self, x):
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.seq_encoder(x)
        x = h_n[-1]
        x = self.classifier(x)
        return x

    def get_instance_config(self) -> dict:
        """
        Retrieves the configuration of the model instance.

        Returns:
            dict: A dictionary containing the model's configuration.
        """
        return {
            "class_name": self.__class__.__name__,
            "sequence_encoder": {
                "n_input_channels": self.n_input_channels,
                "n_seq_encoder_layers": self.n_seq_encoder_layers,
                "hidden_size": self.hidden_size,
                "dropout": self.seq_encoder_dropout,
                "layer_type": self.seq_encoder_layer_type,
            },
            "classifier": {
                "n_classifier_layers": self.n_classifier_layers,
                "n_units_per_classifier_layer": self.n_units_per_classifier_layer,
                "n_classes": self.n_classes,
            },
        }
