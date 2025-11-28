import torch
from diffusers import AutoencoderKL
from workflow_actions.paths import MODEL_READY_DATA_DIR
import torch.nn as nn

from architectures.assemblies.assembly import Assembly
from architectures.model_components.classifier.base_classifier import BaseClassifier


class VAEAssembly(nn.Module, Assembly):
    def __init__(self):
        nn.Module.__init__(self)
        Assembly.__init__(self)
        self.vae = AutoencoderKL.from_pretrained(
            "cvssp/audioldm2",
            subfolder="vae",
        ).eval()

        for p in self.vae.parameters():
            p.requires_grad = False

        self.n_seq_encoder_layers = None
        self.n_units_per_seq_encoder_layer = None
        self.seq_encoder_activation = None
        self.n_embedding_dims = None

        self.seq_encoder_input_features = None
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
        self._infer_vae_output_shape()
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

    def _infer_vae_output_shape(self):
        sample_path = MODEL_READY_DATA_DIR / "train" / "X_0.pt"
        if not sample_path.is_file():
            raise FileNotFoundError(
                f"X_0.pt file from {MODEL_READY_DATA_DIR / 'train'}"
                " is required to infer VAE output shape"
            )
        sample = torch.load(sample_path)
        sample = self._resize_layer(sample)
        sample = self._normalization_layer(sample)
        with torch.no_grad():
            vae_out = self.vae.encode(sample)
            embedding = vae_out.latent_dist.mean
        self.seq_encoder_input_features = embedding.view(embedding.size(0), -1).size(1)

    def forward(self, x):
        x = self.make_embeddings(x)
        x = self.classifier(x)
        return x

    def make_embeddings(self, x):
        x = self._resize_layer(x)
        x = self.input_normalization_layer(x)
        vae_out = self.vae.encode(x)
        embedding = vae_out.latent_dist.mean
        embedding = embedding.view(embedding.size(0), -1)
        return self.seq_encoder(embedding)

    def _get_normalization_layer(self):
        """For abstract class interface consistency."""
        return self._normalization_layer

    @staticmethod
    def _normalization_layer(x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-9)
        x = x * 2 - 1
        return x

    @staticmethod
    def _resize_layer(x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        h, w = x.size(-2), x.size(-1)
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32
        x = torch.nn.functional.interpolate(
            x, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        return x

    def get_instance_config(self) -> dict:
        return {
            "class_name": self.__class__.__name__,
            "sequence_encoder": {
                "n_seq_encoder_layers": self.n_seq_encoder_layers,
                "n_units_per_seq_encoder_layer": self.n_units_per_seq_encoder_layer,
                "seq_encoder_activation": self.seq_encoder_activation,
                "n_embedding_dims": self.n_embedding_dims,
            },
            "classifier": self.get_classifier_config(),
        }
