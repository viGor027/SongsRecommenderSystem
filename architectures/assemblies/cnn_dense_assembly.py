import torch.nn as nn
from architectures.model_components.classifier.base_classifier import BaseClassifier
from architectures.assemblies.cnn_assembly_parent import CnnAssemblyParent
from architectures.model_components.temporal_compressor.convolutional.conv2d_no_skip import (
    Conv2DBlockNoSkip,
)
from architectures.model_components.temporal_compressor.convolutional.conv2d_with_skip import (
    Conv2DBlockWithSkip,
)


class CnnDenseAssembly(nn.Module, CnnAssemblyParent):
    """
    A wrapper for convenient model assembling.

    You must initialize each part of the model before usage:
     '''
        model = CnnDenseAssembly()
        model.init_conv(...)
        model.init_seq_encoder(...)
        model.init_classifier(...)
     '''
    """

    def __init__(self):
        """All the below attributes are set during initialization mentioned in class docstring."""
        nn.Module.__init__(self)
        CnnAssemblyParent.__init__(self)

        self.seq_encoder_input_features = None
        self.n_seq_encoder_layers = None
        self.n_units_per_seq_encoder_layer = None
        self.n_embedding_dims = None

        self.seq_encoder = None

    def init_seq_encoder(
        self,
        n_seq_encoder_layers: int,
        n_units_per_seq_encoder_layer: list[int],
        seq_encoder_activation: str,
        n_embedding_dims: int,
    ):
        self.n_seq_encoder_layers = n_seq_encoder_layers
        self.n_units_per_seq_encoder_layer = n_units_per_seq_encoder_layer
        self.seq_encoder_activation = seq_encoder_activation
        self.n_embedding_dims = n_embedding_dims
        self.seq_encoder = self._build_seq_encoder()

    def _build_seq_encoder(self):
        self._infer_conv_output_shape()
        seq_encoder = BaseClassifier(
            n_layers=self.n_seq_encoder_layers,
            n_input_features=self.seq_encoder_input_features,
            units_per_layer=self.n_units_per_seq_encoder_layer,
            activation=self.seq_encoder_activation,
            n_classes=self.n_embedding_dims,
            sigmoid_output=False,
        )
        return seq_encoder

    def _infer_conv_output_shape(self):
        from workflow_actions.paths import MODEL_READY_DATA_DIR
        import torch

        if self.conv is None:
            raise ValueError(
                "Convolutional part of the network can't be uninitialized."
            )
        if not (MODEL_READY_DATA_DIR / "train" / "X_0.pt").is_file():
            raise FileNotFoundError(
                f"X_0.pt file from {MODEL_READY_DATA_DIR / 'train'}"
                " is required to infer convolution output shape"
            )
        sample = torch.load(MODEL_READY_DATA_DIR / "train" / "X_0.pt")
        if issubclass(self.ConvCls, Conv2DBlockNoSkip) or issubclass(
            self.ConvCls, Conv2DBlockWithSkip
        ):
            sample = sample.unsqueeze(1)
        sample = self.conv(sample)
        self.seq_encoder_input_features = sample.view(sample.size(0), -1).size(1)

    def _classifier_in_features(self) -> int:
        return self.n_embedding_dims

    def forward(self, x):
        x = self.make_embeddings(x)
        x = self.classifier(x)
        return x

    def make_embeddings(self, x):
        if issubclass(self.ConvCls, Conv2DBlockNoSkip) or issubclass(
            self.ConvCls, Conv2DBlockWithSkip
        ):
            x = x.unsqueeze(1)
        x = self.input_normalization_layer(x)
        x = self.conv(x)
        x = x.reshape((x.size(0), -1))
        x = self.seq_encoder(x)
        return x

    def get_instance_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "temporal_compressor": self.get_temporal_compressor_config(),
            "sequence_encoder": {
                "n_seq_encoder_layers": self.n_seq_encoder_layers,
                "n_units_per_seq_encoder_layer": self.n_units_per_seq_encoder_layer,
                "seq_encoder_activation": self.seq_encoder_activation,
                "n_embedding_dims": self.n_embedding_dims,
            },
            "classifier": self.get_classifier_config(),
        }
