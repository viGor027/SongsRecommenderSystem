from torch import nn
import torch
from collections import OrderedDict
from typing import Literal


class Conv2DBaseBlock(nn.Module):
    """
    A convolutional block that processes 2D inputs.
    Block created with this class has 'same' padding on every layer.

    Dimensionality reduction depends on parameters passed to initializer.
    """

    def __init__(
        self,
        block_num: int,
        n_input_channels: int,
        n_layers: int,
        n_filters_per_layer: int,
        kernel_size: int,
        stride: int = 1,  # stride 1 due to `same padding` applied
        activation: Literal["relu", "hardswish"] = "relu",
        reduction_strat: Literal["conv", "max_pool", "avg_pool"] = "conv",
        reduction_kernel_size: int = 2,
        reduction_stride: int = 2,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Notes:
            - block_num indicates the sequential position of this block in the model.
            - dimensionality reduction depends on `reduction_kernel_size` and `reduction_stride`.
        """

        super().__init__()

        self.block_num = block_num

        self.n_input_channels = n_input_channels
        self.n_layers = n_layers
        self.n_filters_per_layer = n_filters_per_layer

        self.kernel_size = kernel_size
        self.stride = stride

        self.reduction_strat = reduction_strat
        self.reduction_kernel_size = reduction_kernel_size
        self.reduction_stride = reduction_stride

        self.dtype = dtype

        activation_map = {"relu": nn.ReLU, "hardswish": nn.Hardswish}
        self.activation = activation_map[activation]

        self.block = self.build_conv_block()

    def build_conv_block(self):
        layers = [
            (
                f"block_{self.block_num}_starting_batch_norm",
                nn.BatchNorm2d(self.n_input_channels),
            )
        ]
        for i in range(self.n_layers):
            in_channels = self.n_filters_per_layer if i != 0 else self.n_input_channels
            layers.append(
                (
                    f"block_{self.block_num}_conv_{i}",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=self.n_filters_per_layer,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        padding="same",
                        dtype=self.dtype,
                    ),
                )
            )
            layers.append((f"block_{self.block_num}_activation_{i}", self.activation()))
            layers.append(
                (
                    f"block_{self.block_num}_batch_norm_{i}",
                    nn.BatchNorm2d(self.n_filters_per_layer),
                )
            )

        # reduces dimensionality
        if self.reduction_strat == "conv":
            in_channels = (
                self.n_filters_per_layer
                if self.n_layers != 0
                else self.n_input_channels
            )
            layers.append(
                (
                    f"block_{self.block_num}_conv_reduce",
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=self.n_filters_per_layer,
                        kernel_size=self.reduction_kernel_size,
                        stride=self.reduction_stride,
                        dtype=self.dtype,
                    ),
                )
            )
        elif self.reduction_strat == "max_pool":
            layers.append(
                (
                    f"block_{self.block_num}_max_pool_reduce",
                    nn.MaxPool2d(
                        kernel_size=self.reduction_kernel_size,
                        stride=self.reduction_stride,
                    ),
                )
            )
        elif self.reduction_strat == "avg_pool":
            layers.append(
                (
                    f"block_{self.block_num}_avg_pool_reduce",
                    nn.AvgPool2d(
                        kernel_size=self.reduction_kernel_size,
                        stride=self.reduction_stride,
                    ),
                )
            )
        layers.append(
            (f"block_{self.block_num}_end_block_activation", self.activation())
        )
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        """Note: PyTorch forward method expects the input to be a batch of samples, even if the batch size is 1."""
        return self.block(x)

    def debug_forward(self, x):
        for name, layer in self.block.named_children():
            print("Name: ", name, " Layer: ", layer)
            x = layer(x)
            print(f"Output shape {x.shape}")
            print()
        return x


if __name__ == "__main__":
    # Usage example
    import torch

    sample_len = 216
    sample_channels = 80

    sample = torch.randn((4, 1, sample_channels, sample_len))

    model = Conv2DBaseBlock(
        block_num=1,
        n_input_channels=1,
        n_layers=2,
        n_filters_per_layer=16,
        kernel_size=2,
    )

    sample = model.debug_forward(sample)
    print("Shape after: ", sample.shape)
    print("Resulting tensor: ")
    print()
