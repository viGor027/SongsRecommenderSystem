from torch import nn
import torch
from collections import OrderedDict
from typing import Literal
from architectures.model_components.temporal_compressor.convolutional.conv_blocks_reduction_parent import (
    ConvBlocksReductionParent,
)


class Conv2DBaseBlock(nn.Module, ConvBlocksReductionParent):
    """
    A convolutional block that applies 2D convolution on inputs.
    Block created with this class has 'same' padding on every layer.
    """

    def __init__(
        self,
        block_num: int,
        n_input_channels: int,
        n_layers: int,
        n_filters_per_layer: int,
        kernel_size: int,
        stride: int = 1,  # stride 1 due to `same` padding applied
        activation: Literal["relu", "hardswish"] = "relu",
        reduction_strat: Literal["conv", "max_pool", "avg_pool"] = "conv",
        reduction_kernel_size: int = 2,
        reduction_stride: int = 2,
        dtype: torch.dtype = torch.float32,
    ):
        nn.Module.__init__(self)

        self.block_num = block_num

        self.n_input_channels = n_input_channels
        self.n_layers = n_layers
        self.n_filters_per_layer = n_filters_per_layer

        self.kernel_size = kernel_size
        self.stride = stride

        activation_map = {"relu": nn.ReLU, "hardswish": nn.Hardswish}
        self.activation = activation_map[activation]

        ConvBlocksReductionParent.__init__(
            self,
            block_type="2d",
            reduction_strat=reduction_strat,
            reduction_kernel_size=reduction_kernel_size,
            reduction_stride=reduction_stride,
            dtype=dtype,
        )

        self.block = self.build_conv_block()

    def build_conv_block(self):
        layers = []
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
                        bias=False,
                        dtype=self.dtype,
                    ),
                )
            )
            layers.append(
                (
                    f"block_{self.block_num}_batch_norm_{i}",
                    nn.BatchNorm2d(self.n_filters_per_layer),
                )
            )
            layers.append((f"block_{self.block_num}_activation_{i}", self.activation()))

        # reduces dimensionality
        layers.append(self.reduction_layer)
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
