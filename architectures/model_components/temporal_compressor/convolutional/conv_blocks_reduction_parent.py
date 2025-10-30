import torch.nn as nn
from typing import Literal
import torch


class ConvBlocksReductionParent:
    """
    This class is not meant to initialize objects.
    Created for code redundancy reduction.

    Note: __init__ of this class is called at the very end of
     __init__ call of child classes (i.e. Conv2DBaseBlock or Conv1DBaseBlock),
     thus self.block_num, self.n_filters_per_layer, self.n_layers and self.n_input_channels
     are accessible.
    """

    def __init__(
        self,
        block_type: Literal["1d", "2d"],
        reduction_strat: Literal["conv", "max_pool", "avg_pool"] = "conv",
        reduction_kernel_size: int = 2,
        reduction_stride: int = 2,
        dtype: torch.dtype = torch.float32,
    ):
        self.reduction_strat = reduction_strat
        self.reduction_kernel_size = reduction_kernel_size
        self.reduction_stride = reduction_stride
        self.dtype = dtype

        base_kwargs = {
            "kernel_size": self.reduction_kernel_size,
            "stride": self.reduction_stride,
        }
        conv_kwargs = {
            "in_channels": (
                self.n_filters_per_layer
                if self.n_layers != 0
                else self.n_input_channels
            ),
            "out_channels": self.n_filters_per_layer,
            "dtype": self.dtype,
        }

        reduction_functions_map = {
            "conv": nn.Conv1d if block_type == "1d" else nn.Conv2d,
            "max_pool": nn.MaxPool1d if block_type == "1d" else nn.MaxPool2d,
            "avg_pool": nn.AvgPool1d if block_type == "1d" else nn.AvgPool2d,
        }
        reduction_func = reduction_functions_map[reduction_strat]

        reduction_layer_kwargs = (
            conv_kwargs | base_kwargs if reduction_strat == "conv" else base_kwargs
        )
        self.reduction_layer = (
            f"block_{self.block_num}_{reduction_strat}_reduce",
            reduction_func(**reduction_layer_kwargs),
        )
