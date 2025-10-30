from torch import nn
import torch
from collections import OrderedDict
from typing import Literal
from architectures.model_components.temporal_compressor.convolutional.conv_blocks_reduction_parent import (
    ConvBlocksReductionParent,
)


class Conv1DBaseBlock(nn.Module, ConvBlocksReductionParent):
    """
    A convolutional block that applies 1D convolution on inputs without dilation by default.
    This class is implemented with causal padding to ensure that future time steps do not influence the output
    at any given time step, as the compressed output is designed to be passed to an RNN for further temporal processing.
    """

    def __init__(
        self,
        block_num: int,
        input_len: int,
        n_input_channels: int,
        n_layers: int,
        n_filters_per_layer: int,
        kernel_size: int,
        stride: int = 1,  # stride 1 due to `same padding` applied
        activation: Literal["relu", "hardswish"] = "relu",
        dilation: bool = False,
        reduction_strat: Literal["conv", "max_pool", "avg_pool"] = "conv",
        reduction_kernel_size: int = 2,
        reduction_stride: int = 2,
        dtype: torch.dtype = torch.float32,
    ):
        nn.Module.__init__(self)

        self.block_num = block_num

        self.input_len = input_len
        self.n_input_channels = n_input_channels
        self.n_layers = n_layers

        self.n_filters_per_layer = n_filters_per_layer
        self.kernel_size = kernel_size
        self.stride = stride

        self.dilation = dilation

        activation_map = {"relu": nn.ReLU, "hardswish": nn.Hardswish}
        self.activation = activation_map[activation]

        ConvBlocksReductionParent.__init__(
            self,
            block_type="1d",
            reduction_strat=reduction_strat,
            reduction_kernel_size=reduction_kernel_size,
            reduction_stride=reduction_stride,
            dtype=dtype,
        )

        self.block = self.build_conv_block()

    def build_conv_block(self):
        layers = [
            (
                f"block_{self.block_num}_starting_batch_norm",
                nn.BatchNorm1d(self.n_input_channels),
            )
        ]

        if dil := self._get_block_dilation():
            dilation = dil
        else:
            dilation = [1 for _ in range(self.n_layers)]

        for i in range(self.n_layers):
            layer_dilation = dilation[i]
            in_channels = self.n_filters_per_layer if i != 0 else self.n_input_channels
            layers.append(
                (
                    f"block_{self.block_num}_padding_{i}",
                    self._get_padding_layer(layer_dilation),
                )
            )
            layers.append(
                (
                    f"block_{self.block_num}_conv_{i}",
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=self.n_filters_per_layer,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=layer_dilation,
                        bias=False,
                        dtype=self.dtype,
                    ),
                )
            )
            layers.append(
                (
                    f"block_{self.block_num}_batch_norm_{i}",
                    nn.BatchNorm1d(self.n_filters_per_layer),
                )
            )
            layers.append((f"block_{self.block_num}_activation_{i}", self.activation()))

        # divides len of time dimension by two
        layers.append(self.reduction_layer)
        return nn.Sequential(OrderedDict(layers))

    def _get_padding_layer(self, dilation: int):
        """Calculates and returns a padding layer to ensure causal padding for the convolution."""
        padding = (
            (self.input_len - 1) * self.stride
            - self.input_len
            + dilation * (self.kernel_size - 1)
            + 1
        )
        return nn.ConstantPad1d((padding, 0), 0)

    def _get_block_dilation(self):
        """
        Determines the dilation values for the convolutional layers in the block.

        Returns:
            list[int]: A list of dilation values for the block if dilation is enabled;
                       otherwise, an empty list.
        """
        if self.dilation:
            return [2**i for i in range(self.n_layers)]
        return []

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
