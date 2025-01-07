import torch
from torch import nn
from collections import OrderedDict


class BaseClassifier(nn.Module):
    """
    A base classifier designed primarily for testing and training other modules.
    """
    def __init__(self, n_layers: int, n_input_features: int,
                 units_per_layer: list[int], n_classes: int):
        """
        - n_layers (int): The number of layers in the model.
          This does not include the final layer, which outputs the probabilities for the classes.
        - n_input_features (int): The number of input features for the first layer.
        - units_per_layer (list[int]): A list specifying the number of units(out_features) in each layer.
        - n_classes (int): The number of output classes for the classifier.
        """

        super().__init__()

        self.n_layers = n_layers
        self.n_input_features = n_input_features
        self.units_per_layer = units_per_layer
        self.n_classes = n_classes

        self.block = self.build()

    def build(self):
        layers = [
            ('dense_layer_0',
             nn.Linear(
                 in_features=self.n_input_features,
                 out_features=self.units_per_layer[0],
                 dtype=torch.float32
             )
             ),
            ('batch_norm_classifier_0', nn.BatchNorm1d(self.units_per_layer[0])),
            ('classifier_activation_0', nn.ReLU())
        ]

        for i in range(self.n_layers-1):
            layers.append(
                (f"dense_layer_{i+1}",
                 nn.Linear(
                     in_features=self.units_per_layer[i],
                     out_features=self.units_per_layer[i+1],
                     dtype=torch.float32
                 )
                 )
            )
            layers.append((f"batch_norm_classifier_{i+1}", nn.BatchNorm1d(self.units_per_layer[i+1])))
            layers.append(
                (f"classifier_activation_{i+1}", nn.ReLU())
            )

        layers.append(
            ("dense_classifier",
             nn.Linear(in_features=self.units_per_layer[self.n_layers-1],
                       out_features=self.n_classes,
                       dtype=torch.float32)
             )
        )
        layers.append(("batch_norm_classifier_end", nn.BatchNorm1d(self.n_classes)))
        layers.append(
            ("classifier_end_activation", nn.Sigmoid())
        )

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


if __name__ == "__main__":
    # Usage example
    n_layers = 3
    n_input_features = 10
    units_per_layer = [32, 64, 128]
    n_classes = 5

    model = BaseClassifier(
        n_layers=n_layers,
        n_input_features=n_input_features,
        units_per_layer=units_per_layer,
        n_classes=n_classes
    )

    # batch of 4 samples with `n_input_features` per sample
    dummy_input = torch.randn(4, n_input_features, dtype=torch.float32)

    #output = model(dummy_input)
    output = model.debug_forward(dummy_input)
