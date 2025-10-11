from torch import nn
import torch
from collections import OrderedDict
from typing import Literal


class Conv1DBaseBlock(nn.Module):
    """
    A convolutional block that processes 1D inputs without dilation by default.
    This class is implemented with causal padding to ensure that future time steps do not influence the output
    at any given time step, as the compressed output is designed to be passed to an RNN for further temporal processing.

    Notes:
        - Dilation in this class is provided to allow the block to be used
          in a convenient way as a component in larger blocks.
        - Every instance of this block will compress the temporal dimension (length of the time axis) by a factor of 2.
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
        """Note: block_num indicates the sequential position of this block in the model."""

        super().__init__()

        self.block_num = block_num

        self.input_len = input_len
        self.n_input_channels = n_input_channels
        self.n_layers = n_layers

        self.n_filters_per_layer = n_filters_per_layer
        self.kernel_size = kernel_size
        self.stride = stride

        self.dilation = dilation

        self.reduction_strat = reduction_strat
        self.reduction_kernel_size = reduction_kernel_size
        self.reduction_stride = reduction_stride

        self.padding_left = (
            (input_len - 1) * stride - input_len + 1 * (kernel_size - 1) + 1
        )
        self.dtype = dtype

        activation_map = {"relu": nn.ReLU, "hardswish": nn.Hardswish}
        self.activation = activation_map[activation]

        self.block = self.build_conv_block()

    def build_conv_block(self):
        layers = []

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
                        dtype=self.dtype,
                    ),
                )
            )
            layers.append((f"block_{self.block_num}_activation_{i}", self.activation()))

        # divides len of time dimension by two
        if self.reduction_strat == "conv":
            in_channels = (
                self.n_filters_per_layer
                if self.n_layers != 0
                else self.n_input_channels
            )
            layers.append(
                (
                    f"block_{self.block_num}_conv_reduce",
                    nn.Conv1d(
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
                    nn.MaxPool1d(
                        kernel_size=self.reduction_kernel_size,
                        stride=self.reduction_stride,
                    ),
                )
            )
        elif self.reduction_strat == "avg_pool":
            layers.append(
                (
                    f"block_{self.block_num}_avg_pool_reduce",
                    nn.AvgPool1d(
                        kernel_size=self.reduction_kernel_size,
                        stride=self.reduction_stride,
                    ),
                )
            )
        layers.append(
            (f"block_{self.block_num}_end_block_activation", self.activation())
        )
        return nn.Sequential(OrderedDict(layers))

    def _get_padding_layer(self, dilation: int):
        """
        Calculates and returns a padding layer to ensure causal padding for the convolution.

        Args:
            dilation (int): The dilation value for the corresponding convolution layer.

        Returns:
            nn.ConstantPad2d: A padding layer with the calculated padding size.
        """
        padding = (
            (self.input_len - 1) * self.stride
            - self.input_len
            + dilation * (self.kernel_size - 1)
            + 1
        )
        return nn.ConstantPad2d((padding, 0, 0, 0), 0)

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


if __name__ == "__main__":
    # Usage example
    import torch

    sample_len = 200
    sample_channels = 80

    sample = torch.randn((4, sample_channels, sample_len))

    model = Conv1DBaseBlock(
        block_num=1,
        input_len=sample_len,
        n_input_channels=sample_channels,
        kernel_size=2,
        stride=1,
        n_filters_per_layer=32,
        n_layers=2,
        dilation=True,
    )

    # sample = model(sample)
    sample = model.debug_forward(sample)
    print("Shape after: ", sample.shape)
    print("Resulting tensor: ")
    print()
