from torch import nn
from architectures.model_components.temporal_compressor.convolutional.conv2d_base_block import (
    Conv2DBaseBlock,
)
import torch
from typing import Literal


class Conv2DBlockWithSkip(nn.Module):
    """
    A convolutional block that processes 2D without skip connections.

    Notes:
        - Every instance of this block will compress the spatial dimensions by a factor of 2.
    """

    def __init__(
        self,
        block_num: int,
        n_input_channels: int,
        n_layers: int,
        n_filters_per_layer: int,
        n_filters_skip: int,
        kernel_size: int,
        stride: int = 1,
        activation: Literal["relu", "hardswish"] = "relu",
        reduction_strat: Literal["conv", "max_pool", "avg_pool"] = "conv",
        reduction_kernel_size: int = 2,
        reduction_stride: int = 2,
        dtype: torch.dtype = torch.float32,
    ):
        """Note: input_len is not used and is present for API consistency"""
        super().__init__()

        self.skip_reducing_conv = nn.Conv2d(
            in_channels=n_input_channels,
            out_channels=n_filters_skip,
            kernel_size=reduction_kernel_size,
            stride=reduction_stride,
            dtype=dtype,
        )

        self.block = Conv2DBaseBlock(
            block_num=block_num,
            n_input_channels=n_input_channels,
            n_layers=n_layers,
            n_filters_per_layer=n_filters_per_layer,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            reduction_strat=reduction_strat,
            reduction_kernel_size=reduction_kernel_size,
            reduction_stride=reduction_stride,
            dtype=dtype,
        )

        self.block_num = block_num

    def forward(self, x):
        """Note: PyTorch forward method expects the input to be a batch of samples, even if the batch size is 1."""
        x_reduced = self.skip_reducing_conv(x)
        x = self.block(x)
        out = torch.cat((x, x_reduced), dim=1)
        return out

    def debug_forward(self, x):
        x_reduced = self.skip_reducing_conv(x)
        for name, layer in self.block.named_children():
            print("Name: ", name, " Layer: ", layer)
            x = layer(x)
            print(f"Output shape {x.shape}")
            print()
        print(f"block_{self.block_num}_reducing")
        print(f"Output shape of reducing layer {x_reduced.shape}")
        out = torch.cat((x, x_reduced), dim=1)
        return out


if __name__ == "__main__":
    # Usage example
    import torch

    sample_len = 216
    sample_channels = 80

    sample = torch.randn((4, 1, sample_channels, sample_len))

    model = Conv2DBlockWithSkip(
        block_num=1,
        n_input_channels=1,
        n_layers=2,
        n_filters_per_layer=16,
        n_filters_skip=5,
        kernel_size=3,
        reduction_strat="max_pool",
    )
    sample = model.debug_forward(sample)
    print("Shape after: ", sample.shape)
    print("Resulting tensor: ")
    print()
