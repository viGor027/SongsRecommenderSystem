import torch.nn as nn
from architectures.assemblies.cnn_assembly_parent import CnnAssemblyParent
from typing import Literal


class CnnRnnDenseAssembly(nn.Module, CnnAssemblyParent):
    """
    A wrapper for convenient model assembling.

    You must initialize each part of the model before usage:
    '''
        model = CnnRnnDenseAssembly()
        model.init_conv(...)
        model.init_seq_encoder(...)
        model.init_classifier(...)
    '''

    **IMPORTANT NOTE: Do not use with Conv2D convolutional blocks**
    """

    def __init__(self):
        """All the below attributes are set during initialization mentioned in class docstring."""
        nn.Module.__init__(self)
        CnnAssemblyParent.__init__(self)

        self.n_seq_encoder_layers = None
        self.hidden_size = None
        self.seq_encoder_dropout = None
        self.seq_encoder_layer_type = None

        self.seq_encoder = None

        self.forward_func = None

    def init_seq_encoder(
        self,
        n_seq_encoder_layers: int,
        hidden_size: int,
        dropout: float,
        layer_type: Literal["gru", "lstm"],
    ):
        self.n_seq_encoder_layers = n_seq_encoder_layers
        self.hidden_size = hidden_size
        self.seq_encoder_dropout = dropout
        self.seq_encoder_layer_type = layer_type
        self.seq_encoder = self._build_seq_encoder()

        self.forward_func = (
            self._forward_gru if layer_type == "gru" else self._forward_lstm
        )

    def _build_seq_encoder(self):
        n_filters_in_last_skip = (
            self.n_filters_per_skip[-1] if self.n_filters_per_skip is not None else 0
        )
        if self.seq_encoder_layer_type == "gru":
            return nn.Sequential(
                nn.GRU(
                    input_size=self.n_filters_per_block[-1] + n_filters_in_last_skip,
                    hidden_size=self.hidden_size,
                    dropout=self.seq_encoder_dropout,
                    num_layers=self.n_seq_encoder_layers,
                    batch_first=True,
                )
            )
        else:
            return nn.Sequential(
                nn.LSTM(
                    input_size=self.n_filters_per_block[-1] + n_filters_in_last_skip,
                    hidden_size=self.hidden_size,
                    dropout=self.seq_encoder_dropout,
                    num_layers=self.n_seq_encoder_layers,
                    batch_first=True,
                )
            )

    def _classifier_in_features(self) -> int:
        return self.hidden_size

    def forward(self, x):
        x = self.input_normalization_layer(x)
        x = self.forward_func(x)
        return x

    def _forward_gru(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, h_n = self.seq_encoder(x)
        x = h_n[-1]
        x = self.classifier(x)
        return x

    def _forward_lstm(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.seq_encoder(x)
        x = h_n[-1]
        x = self.classifier(x)
        return x

    def get_instance_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "temporal_compressor": self.get_temporal_compressor_config(),
            "sequence_encoder": {
                "n_seq_encoder_layers": self.n_seq_encoder_layers,
                "hidden_size": self.hidden_size,
                "dropout": self.seq_encoder_dropout,
                "layer_type": self.seq_encoder_layer_type,
            },
            "classifier": self.get_classifier_config(),
        }
