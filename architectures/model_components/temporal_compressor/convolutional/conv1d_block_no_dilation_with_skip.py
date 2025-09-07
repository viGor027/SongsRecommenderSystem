from torch import nn
from architectures.model_components.temporal_compressor.convolutional.conv1d_block_no_dilation_no_skip import (
    Conv1DBlockNoDilationNoSkip,
)
from typing import Literal


class Conv1DBlockNoDilationWithSkip(nn.Module):
    """
    A convolutional block that processes 1D inputs without dilation, incorporating skip connection.

    This block combines the output of a convolutional block (`Conv1DBlockNoDilationNoSkip`) with a halved version
    of its input. The halving is achieved using a convolutional layer with `stride=2` and `kernel_size=2`.

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
        stride: int,
        reduction_strat: Literal["conv", "max_pool", "avg_pool"] = "conv",
    ):
        """
        Notes:
            - block_num indicates the sequential position of this block in the model.
            - input_len is a Length of the input's temporal dimension, corresponding to L_in in temporal_compressor/note.md.
            - n_input_channels is equal to n_mels if this is the first block in a model.
        """
        super().__init__()

        self.block_num = block_num

        self.skip_halving_conv = nn.Conv1d(
            in_channels=n_input_channels,
            out_channels=n_filters_skip,
            kernel_size=2,
            stride=2,
            dtype=torch.float32,
        )

        self.block = Conv1DBlockNoDilationNoSkip(
            block_num=block_num,
            input_len=input_len,
            n_input_channels=n_input_channels,
            n_layers=n_layers,
            n_filters_per_layer=n_filters_per_layer,
            kernel_size=kernel_size,
            stride=stride,
            n_filters_skip=-1,
            reduction_strat=reduction_strat,
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


if __name__ == "__main__":
    # Usage example
    import torch

    sample_len = 200
    sample_channels = 80

    sample = torch.randn((4, sample_channels, sample_len))

    model = Conv1DBlockNoDilationWithSkip(
        block_num=1,
        input_len=sample_len,
        n_input_channels=sample_channels,
        kernel_size=2,
        stride=1,
        n_filters_per_layer=64,
        n_filters_skip=16,
        n_layers=2,
        reduction_strat="conv",
    )
    sample = model.debug_forward(sample)
    print("Shape after: ", sample.shape)
    print("Resulting tensor: ")
    print()
