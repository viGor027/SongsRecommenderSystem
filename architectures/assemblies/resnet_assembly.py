import torch
import torch.nn as nn
from torchvision import models

from architectures.assemblies.assembly import Assembly
from architectures.model_components.classifier.base_classifier import BaseClassifier
from typing import Literal


class ResNetAssembly(nn.Module, Assembly):
    def __init__(
        self,
        backbone_name: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101"
        ] = "resnet18",
        weights: Literal["DEFAULT", "IMAGENET1K_V1", "IMAGENET1K_V2", None] = None,
        freeze_backbone: bool = True,
    ):
        nn.Module.__init__(self)
        Assembly.__init__(self)

        if backbone_name not in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            raise ValueError(
                "backbone_name must be 'resnet18', 'resnet34', 'resnet50' or 'resnet101'."
            )

        if weights == "IMAGENET1K_V2" and backbone_name not in [
            "resnet50",
            "resnet101",
        ]:
            raise ValueError(
                "Weights 'IMAGENET1K_V2' are only supported for backbone 'resnet50' and 'resnet101'."
            )

        self.backbone_name = backbone_name
        self.weights = weights
        self.freeze_backbone = freeze_backbone

        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=self.weights)
            feature_dim = 512
        elif backbone_name == "resnet34":
            backbone = models.resnet34(weights=self.weights)
            feature_dim = 512
        elif backbone_name == "resnet50":
            backbone = models.resnet50(weights=self.weights)
            feature_dim = 2048
        else:  # "resnet101"
            backbone = models.resnet101(weights=self.weights)
            feature_dim = 2048

        self.feature_dim = feature_dim
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.n_seq_encoder_layers = None
        self.n_units_per_seq_encoder_layer = None
        self.seq_encoder_activation = None
        self.n_embedding_dims = None

        self.seq_encoder_input_features = self.feature_dim
        self.seq_encoder = None

        self.input_normalization_layer = self._get_normalization_layer()

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
        return BaseClassifier(
            n_layers=self.n_seq_encoder_layers,
            n_input_features=self.seq_encoder_input_features,
            units_per_layer=self.n_units_per_seq_encoder_layer,
            activation=self.seq_encoder_activation,
            n_classes=self.n_embedding_dims,
            sigmoid_output=False,
        )

    def _classifier_in_features(self) -> int:
        return self.n_embedding_dims

    def forward(self, x):
        x = self.make_embeddings(x)
        x = self.classifier(x)
        return x

    def make_embeddings(self, x):
        x = self._resize_layer(x)
        x = self.input_normalization_layer(x)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        return self.seq_encoder(feats)

    def _get_normalization_layer(self):
        return self._normalization_layer

    @staticmethod
    def _normalization_layer(x):
        x_min = x.amin(dim=[1, 2, 3], keepdim=True)
        x_max = x.amax(dim=[1, 2, 3], keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-9)
        x = x * 2 - 1
        return x

    @staticmethod
    def _resize_layer(x):
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )
        return x

    def get_instance_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "backbone_name": self.backbone_name,
            "weights": self.weights,
            "freeze_backbone": self.freeze_backbone,
            "sequence_encoder": {
                "n_seq_encoder_layers": self.n_seq_encoder_layers,
                "n_units_per_seq_encoder_layer": self.n_units_per_seq_encoder_layer,
                "seq_encoder_activation": self.seq_encoder_activation,
                "n_embedding_dims": self.n_embedding_dims,
            },
            "classifier": self.get_classifier_config(),
        }
