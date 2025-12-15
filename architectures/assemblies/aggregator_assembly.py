import torch.nn as nn
from typing import Literal
from typing import TYPE_CHECKING, Union
from architectures.model_components.classifier.base_classifier import BaseClassifier

if TYPE_CHECKING:
    from resnet_assembly import ResNetAssembly
    from dense_assembly import DenseAssembly


class AggregatorAssembly(nn.Module):
    def __init__(
        self,
        embedding_model: Union["ResNetAssembly", "DenseAssembly"],
        aggregator_type: Literal["average", "lstm"],
        trainable_aggregator: bool = True,
    ):
        nn.Module.__init__(self)

        self.embedding_model = embedding_model
        self.embedding_model.requires_grad_(False)

        self.aggregator_type = aggregator_type
        self.trainable_aggregator = trainable_aggregator

        self.aggregator = self._build_aggregator(aggregator_type)

        self.n_classes = None
        self.classifier = None

    def train(self, mode: bool = True):
        super().train(mode)
        self.embedding_model.eval()
        return self

    def forward(self, x):
        x = self.aggregate_fragments(x)
        out = self.classifier(x)
        return out

    def aggregate_fragments(self, x):
        """x has shape of [n_fragments, n_mels, len] contain single song fragments"""
        embedded_fragments = self.embedding_model.make_embeddings(
            x
        )  # shape [n_fragments, n_embedding_dims]
        embedded_fragments = embedded_fragments.unsqueeze(
            0
        )  # shape [1, n_fragments, n_embedding_dims]
        aggregated_fragments = self.aggregator(embedded_fragments)
        return aggregated_fragments

    def init_classifier(
        self,
        n_classes: int,
    ):
        self.n_classes = n_classes
        self.classifier = self._build_classifier()

    def _build_classifier(self):
        classifier = BaseClassifier(
            n_layers=1,
            n_input_features=self.embedding_model.n_embedding_dims,
            units_per_layer=[],
            activation=None,
            n_classes=self.n_classes,
            sigmoid_output=True,
        )
        return classifier

    def _build_aggregator(self, aggregator_type):
        self._aggregator_layer = {
            "lstm": nn.LSTM(
                input_size=self.embedding_model.n_embedding_dims,
                hidden_size=self.embedding_model.n_embedding_dims,
                dropout=0,
                num_layers=1,
                batch_first=True,
            ),
            "average": None,
        }[aggregator_type]

        if not self.trainable_aggregator and aggregator_type in ["lstm"]:
            for p in self._aggregator_layer.parameters():
                p.requires_grad = False

        return {
            "lstm": self._lstm_pass,
            "average": self._average_pass,
        }[aggregator_type]

    def _lstm_pass(self, x):
        _, (h_n, _) = self._aggregator_layer(x)
        x = h_n[-1]
        return x

    def _average_pass(self, x):
        x = x.mean(dim=1)
        return x

    def get_instance_config(self):
        return {
            "class_name": self.__class__.__name__,
            "embedding_model": self.embedding_model.__class__.__name__,
            "aggregator_type": self.aggregator_type,
            "trainable_aggregator": self.trainable_aggregator,
            "classifier": {
                "n_classes": self.n_classes,
            },
        }
