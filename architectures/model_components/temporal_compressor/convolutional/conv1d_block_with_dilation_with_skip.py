from torch import nn
from architectures.model_components.temporal_compressor.convolutional.conv1d_block_with_dilation_no_skip import (
    Conv1DBlockWithDilationNoSkip,
)
from typing import Literal
import torch


class Conv1DBlockWithDilationWithSkip(nn.Module):
    """
    A convolutional block that processes 1D inputs incorporating dilation and skip connection.

    This block combines the output of a convolutional block (`Conv1DBlockWithDilationNoSkip`) with a halved version
    of its input.

    This class is implemented with causal padding(look at Conv1DBaseBlock implementation for further explanation).

    Notes:
        - The `skip_halving_conv` layer reduces the temporal dimension (time axis length) by a factor of 2.
        - Every instance of this block will compress the temporal dimension (length of the time axis) by a factor of 2.
    """

    def __init__(
        self,
        block_num: int,
        input_len: int,
        n_input_channels: int,
        n_layers: int,
        n_filters_per_layer: int,
        n_filters_skip: int,
        kernel_size: int,
        stride: int = 1,  # stride 1 due to `same padding` applied
        activation: Literal["relu", "hardswish"] = "relu",
        reduction_strat: Literal["conv", "max_pool", "avg_pool"] = "conv",
        reduction_kernel_size: int = 2,
        reduction_stride: int = 2,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.block_num = block_num

        self.skip_halving_conv = nn.Conv1d(
            in_channels=n_input_channels,
            out_channels=n_filters_skip,
            kernel_size=reduction_kernel_size,
            stride=reduction_stride,
            dtype=dtype,
        )

        self.block = Conv1DBlockWithDilationNoSkip(
            block_num=block_num,
            input_len=input_len,
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

    def forward(self, x):
        """Note: PyTorch forward method expects the input to be a batch of samples, even if the batch size is 1."""
        x_halved = self.skip_halving_conv(x)
        x = self.block(x)
        out = torch.cat((x, x_halved), dim=1)
        return out

    def debug_forward(self, x):
        x_halved = self.skip_halving_conv(x)
        for name, layer in self.block.named_children():
            print("Name: ", name, " Layer: ", layer)
            x = layer(x)
            print(f"Output shape {x.shape}")
            print()

        print(f"block_{self.block_num}_halving")
        print(f"Output shape of halving layer {x_halved.shape}")
        out = torch.cat((x, x_halved), dim=1)
        return out
