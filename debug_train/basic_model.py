import torch
import torch.nn as nn
from collections import OrderedDict


class BasicModel(nn.Module):
    def __init__(self, n_input_features: int, n_classes: int):
        super().__init__()

        self.n_input_features = n_input_features
        self.n_classes = n_classes

        self.block = self.build()

    def build(self):
        layers = [
                    ('dense_0', nn.Linear(
                        in_features=self.n_input_features,
                        out_features=self.n_classes,
                        dtype=torch.float16)),
                    ('activation_0', nn.Sigmoid())
                ]
        return nn.Sequential(
            OrderedDict(
                layers
            )
        )

    def forward(self, x):
        """Note: PyTorch forward method expects the input to be a batch of samples, even if the batch size is 1."""
        return self.block(x)

    def debug_forward(self, x):
        for name, layer in self.block.named_children():
            print("Name: ", name, " Layer: ", layer)
            print(f'Contains NaNs before layer: {torch.isnan(x).any()}')
            # if 'batch' not in name and 'activation' not in name:
            #     print("Layer params:")
            #     print(layer.weight)
            #     print(layer.bias)
            x = layer(x)
            print(f'Output shape {x.shape}')
            print(f'Contains NaNs after layer: {torch.isnan(x).any()}')
            print()
        return x
