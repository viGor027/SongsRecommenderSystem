from architectures.assemblies.assembly import Assembly
from typing import Literal
import torch.nn as nn
from functools import partial


class CnnAssemblyParent(Assembly):
    def __init__(self):
        """All the below attributes are set during initialization of child assemblies."""
        super().__init__()

        self.ConvCls = None
        self.n_blocks = None
        self.n_layers_per_block = None
        self.n_filters_per_block = None
        self.n_filters_per_skip = None
        self.reduction_strat = None
        self.reduction_kernel_size = None
        self.reduction_stride = None
        self.input_len = None
        self.conv_activation = None
        self.n_input_channels = None

        self.conv = None

    def init_conv(
        self,
        ConvCls,
        n_blocks: int,
        n_input_channels: int,
        n_layers_per_block: list[int],
        n_filters_per_block: list[int],
        n_filters_per_skip: list[int] | None = None,
        input_len: int | None = None,
        conv_activation: Literal["relu", "hardswish"] = "relu",
        reduction_strat: Literal["conv", "max_pool", "avg_pool"] = "conv",
        reduction_kernel_size: int = 2,
        reduction_stride: int = 2,
    ):
        self.ConvCls = ConvCls
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.n_filters_per_block = n_filters_per_block
        self.n_filters_per_skip = n_filters_per_skip
        self.reduction_strat = reduction_strat
        self.reduction_kernel_size = reduction_kernel_size
        self.reduction_stride = reduction_stride
        self.input_len = input_len
        self.conv_activation = conv_activation
        self.n_input_channels = n_input_channels

        self.conv = self._build_conv()

    def _build_conv(self):
        """
        Builds temporal compressor based on configuration passed to init_conv.

        Returns:
            nn.Sequential: Sequential container of convolutional blocks.
        """
        blocks = []
        for i in range(self.n_blocks):
            blocks.append(self._build_single_block(i=i))
        return nn.Sequential(*blocks)

    def _build_single_block(self, i: int):
        """Takes care on passing or skipping `input_len` and `n_filters_skip` to ConvCls."""
        n_filters_skip_from_prev_block = (
            self.n_filters_per_skip[i - 1]
            if self.n_filters_per_skip is not None and i > 0
            else 0
        )
        n_input_channels = (
            n_filters_skip_from_prev_block + self.n_filters_per_block[i - 1]
            if i > 0
            else self.n_input_channels
        )
        partially_initialized_ConvCls = partial(
            self.ConvCls,
            block_num=i,
            n_input_channels=n_input_channels,
            n_layers=self.n_layers_per_block[i]
            if i != 0
            else self.n_layers_per_block[0],
            n_filters_per_layer=self.n_filters_per_block[i]
            if i != 0
            else self.n_filters_per_block[0],
            activation=self.conv_activation,
            reduction_strat=self.reduction_strat,
            reduction_kernel_size=self.reduction_kernel_size,
            reduction_stride=self.reduction_stride,
            kernel_size=2,
            stride=1,
        )

        filtered_optional_kwargs = (
            {"input_len": self.input_len // 2**i} if self.input_len is not None else {}
        )
        filtered_optional_kwargs = (
            filtered_optional_kwargs | {"n_filters_skip": self.n_filters_per_skip[i]}
            if self.n_filters_per_skip is not None
            else filtered_optional_kwargs
        )
        return partially_initialized_ConvCls(**filtered_optional_kwargs)

    def get_temporal_compressor_config(self) -> dict:
        # Note:
        # This step-by-step dict construction is intentional — it preserves the key order
        # in the returned config, matching the original initialization order for readability.
        temporal_compressor_base_cfg = {
            "ConvCls": self.ConvCls.__name__,
        }
        temporal_compressor_base_cfg = (
            temporal_compressor_base_cfg
            if self.input_len is None
            else temporal_compressor_base_cfg | {"input_len": self.input_len}
        )
        temporal_compressor_base_cfg = temporal_compressor_base_cfg | {
            "n_input_channels": self.n_input_channels,
            "n_blocks": self.n_blocks,
            "n_layers_per_block": self.n_layers_per_block,
            "n_filters_per_block": self.n_filters_per_block,
        }
        temporal_compressor_base_cfg = (
            temporal_compressor_base_cfg
            if self.n_filters_per_skip is None
            else temporal_compressor_base_cfg
            | {"n_filters_per_skip": self.n_filters_per_skip}
        )
        temporal_compressor_base_cfg = temporal_compressor_base_cfg | {
            "reduction_strat": self.reduction_strat,
            "conv_activation": self.conv_activation,
        }
        return temporal_compressor_base_cfg

    def forward(self, x):
        raise NotImplementedError(
            "CnnAssemblyParent class is not meant to initialize objects."
        )

    def get_instance_config(self) -> dict:
        raise NotImplementedError(
            "CnnAssemblyParent class is not meant to initialize objects."
        )

    def _classifier_in_features(self) -> int:
        raise NotImplementedError(
            "CnnAssemblyParent class is not meant to initialize objects."
        )
