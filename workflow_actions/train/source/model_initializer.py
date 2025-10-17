from architectures.assemblies.cnn_dense_assembly import CnnDenseAssembly
from architectures.assemblies.rnn_dense_assembly import RnnDenseAssembly
from architectures.assemblies.cnn_rnn_dense_assembly import CnnRnnDenseAssembly
from architectures.model_components.temporal_compressor.convolutional.conv1d_block_with_dilation_with_skip import (
    Conv1DBlockWithDilationWithSkip,
)
from architectures.model_components.temporal_compressor.convolutional.conv1d_block_with_dilation_no_skip import (
    Conv1DBlockWithDilationNoSkip,
)
from architectures.model_components.temporal_compressor.convolutional.conv1d_block_no_dilation_with_skip import (
    Conv1DBlockNoDilationWithSkip,
)
from architectures.model_components.temporal_compressor.convolutional.conv1d_block_no_dilation_no_skip import (
    Conv1DBlockNoDilationNoSkip,
)
from architectures.model_components.temporal_compressor.convolutional.conv2d_with_skip import (
    Conv2DBlockWithSkip,
)
from architectures.model_components.temporal_compressor.convolutional.conv2d_no_skip import (
    Conv2DBlockNoSkip,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from architectures.assemblies.assembly import Assembly


class ModelInitializer:
    """
    Responsible for initializing Assembly object based on config.
    """
    def __init__(self):
        self.assembly_map = {
            "CnnDenseAssembly": CnnDenseAssembly,
            "RnnDenseAssembly": RnnDenseAssembly,
            "CnnRnnDenseAssembly": CnnRnnDenseAssembly,
        }

        self.conv_cls_map = {
            "Conv1DBlockWithDilationWithSkip": Conv1DBlockWithDilationWithSkip,
            "Conv1DBlockWithDilationNoSkip": Conv1DBlockWithDilationNoSkip,
            "Conv1DBlockNoDilationWithSkip": Conv1DBlockNoDilationWithSkip,
            "Conv1DBlockNoDilationNoSkip": Conv1DBlockNoDilationNoSkip,
            "Conv2DBlockWithSkip": Conv2DBlockWithSkip,
            "Conv2DBlockNoSkip": Conv2DBlockNoSkip,
        }

        self.initializer_map = {
            "CnnDenseAssembly": self._cnn_assembly_init,
            "RnnDenseAssembly": self._rnn_dense_assembly_init,
            "CnnRnnDenseAssembly": self._cnn_assembly_init,
        }

    def get_pretrained_torch_model(self):
        raise NotImplementedError("Implement this method")

    def get_model_from_pretrained(self):
        raise NotImplementedError("Implement this method")

    def get_model_assembly(self, assembly_config: dict) -> "Assembly":
        assembly_class_name = assembly_config.get("class_name", "")
        if not assembly_class_name:
            raise KeyError("assembly_config doesn't contain `class_name` key")

        return self.initializer_map[assembly_class_name](
            assembly_config, assembly_class_name
        )

    def _rnn_dense_assembly_init(
        self, assembly_config: dict, assembly_class_name: str
    ) -> "Assembly":
        sequence_encoder_cfg = assembly_config.get("sequence_encoder", {})
        classifier_cfg = assembly_config.get("classifier", {})
        if not sequence_encoder_cfg:
            raise KeyError("assembly_config doesn't contain `sequence_encoder` key")
        if not classifier_cfg:
            raise KeyError("assembly_config doesn't contain `classifier` key")
        assembly = self.assembly_map[assembly_class_name]()
        assembly.init_seq_encoder(**sequence_encoder_cfg)
        assembly.init_classifier(**classifier_cfg)
        return assembly

    def _cnn_assembly_init(
        self, assembly_config: dict, assembly_class_name: str
    ) -> "Assembly":
        """Function is suitable to initialize both CnnDenseAssembly and CnnRnnDenseAssembly assemblies."""
        temporal_compressor_cfg = assembly_config.get("temporal_compressor", {})
        sequence_encoder_cfg = assembly_config.get("sequence_encoder", {})
        classifier_cfg = assembly_config.get("classifier", {})
        if not temporal_compressor_cfg:
            raise KeyError("assembly_config doesn't contain `temporal_compressor` key")
        if not sequence_encoder_cfg:
            raise KeyError("assembly_config doesn't contain `sequence_encoder` key")
        if not classifier_cfg:
            raise KeyError("assembly_config doesn't contain `classifier` key")
        assembly = self.assembly_map[assembly_class_name]()
        temporal_compressor_cfg["ConvCls"] = self.conv_cls_map[
            temporal_compressor_cfg["ConvCls"]
        ]
        assembly.init_conv(**temporal_compressor_cfg)
        assembly.init_seq_encoder(**sequence_encoder_cfg)
        assembly.init_classifier(**classifier_cfg)
        return assembly
