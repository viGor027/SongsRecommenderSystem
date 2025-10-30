import torch
from torch import nn
from collections import OrderedDict
from typing import Literal


class BaseClassifier(nn.Module):
    """
    A base classifier that can be used either as a standalone model or as part of an assembly.
    """

    def __init__(
        self,
        n_layers: int,
        n_input_features: int,
        units_per_layer: list[int],
        n_classes: int,
        activation: Literal["relu", "hardswish"] | None = "relu",
        sigmoid_output: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        if len(units_per_layer) != n_layers - 1:
            raise ValueError(
                (
                    f"units_per_layer needs to contain exactly {n_layers - 1}"
                    f" (n_layers-1) values but contains {len(units_per_layer)} values."
                )
            )

        super().__init__()

        self.n_layers = n_layers
        self.n_input_features = n_input_features
        self.units_per_layer = units_per_layer
        self.n_classes = n_classes
        self.sigmoid_output = sigmoid_output
        self.dtype = dtype

        activation_map = {"relu": nn.ReLU, "hardswish": nn.Hardswish}
        self.activation = activation_map[activation] if activation is not None else None

        self.block = self.build()

    def build(self):
        layers = []

        for i in range(self.n_layers):
            is_not_last = i != self.n_layers - 1
            in_features = (
                self.n_input_features if i == 0 else self.units_per_layer[i - 1]
            )
            out_features = self.units_per_layer[i] if is_not_last else self.n_classes

            layers.append(
                (
                    f"dense_layer_{i}",
                    nn.Linear(
                        in_features=in_features,
                        out_features=out_features,
                        bias=False,
                        dtype=self.dtype,
                    ),
                )
            )
            batch_norm = (
                [
                    (
                        f"batch_norm_classifier_{i}",
                        nn.BatchNorm1d(num_features=out_features),
                    )
                ]
                if is_not_last
                else []
            )
            layers.extend(batch_norm)
            activation = (
                [(f"classifier_activation_{i}", self.activation())]
                if self.activation is not None and is_not_last
                else []
            )
            layers.extend(activation)

        if self.sigmoid_output:
            layers.append(("classifier_end_activation", nn.Sigmoid()))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        """Note: PyTorch forward method expects the input to be a batch of samples, even if the batch size is 1."""
        return self.block(x)

    def debug_forward(self, x):
        for name, layer in self.block.named_children():
            print("Name: ", name, " Layer: ", layer)
            print(f"Contains NaNs before layer: {torch.isnan(x).any()}")
            x = layer(x)
            print(f"Output shape {x.shape}")
            print(f"Contains NaNs after layer: {torch.isnan(x).any()}")
            print()
        return x
