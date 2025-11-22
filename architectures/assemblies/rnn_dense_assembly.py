import torch.nn as nn
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
        nn.Module.__init__(self)
        Assembly.__init__(self)

        self.n_input_channels = None
        self.n_seq_encoder_layers = None
        self.hidden_size = None
        self.seq_encoder_dropout = None
        self.seq_encoder_layer_type = None

        self.seq_encoder = None

        self.forward_rec = None

        self.input_normalization_layer = None

    def init_seq_encoder(
        self,
        n_input_channels: int,
        n_seq_encoder_layers: int,
        hidden_size: int,
        dropout: float,
        layer_type: Literal["gru", "lstm"],
    ):
        self.n_input_channels = n_input_channels
        self.n_seq_encoder_layers = n_seq_encoder_layers
        self.hidden_size = hidden_size
        self.seq_encoder_dropout = dropout
        self.seq_encoder_layer_type = layer_type
        self.seq_encoder = self._build_seq_encoder()

        self.forward_rec = (
            self._forward_gru if layer_type == "gru" else self._forward_lstm
        )
        self.input_normalization_layer = self._get_normalization_layer()

    def _build_seq_encoder(self):
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

    def _classifier_in_features(self) -> int:
        return self.hidden_size

    def _get_normalization_layer(self):
        return nn.BatchNorm1d(num_features=self.n_input_channels)

    def forward(self, x):
        x = self.make_embeddings(x)
        x = self.classifier(x)
        return x

    def make_embeddings(self, x):
        x = self.input_normalization_layer(x)
        x = self.forward_rec(x)
        return x

    def _forward_gru(self, x):
        x = x.permute(0, 2, 1)
        _, h_n = self.seq_encoder(x)
        x = h_n[-1]
        return x

    def _forward_lstm(self, x):
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.seq_encoder(x)
        x = h_n[-1]
        return x

    def get_instance_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "sequence_encoder": {
                "n_input_channels": self.n_input_channels,
                "n_seq_encoder_layers": self.n_seq_encoder_layers,
                "hidden_size": self.hidden_size,
                "dropout": self.seq_encoder_dropout,
                "layer_type": self.seq_encoder_layer_type,
            },
            "classifier": self.get_classifier_config(),
        }
