from architectures import (
    AggregatorAssembly,
    CnnRnnDenseAssembly,
    CnnDenseAssembly,
    RnnDenseAssembly,
    DenseAssembly,
    ResNetAssembly,
    Conv1DBlockWithDilationNoSkip,
    Conv1DBlockNoDilationNoSkip,
    Conv1DBlockWithDilationWithSkip,
    Conv1DBlockNoDilationWithSkip,
    Conv2DBlockNoSkip,
    Conv2DBlockWithSkip,
)
from typing import TYPE_CHECKING, Union
import torch
from workflow_actions.paths import TRAINED_MODELS_DIR

if TYPE_CHECKING:
    from architectures.assemblies.assembly import Assembly


class ModelInitializer:
    """
    Responsible for initializing Assembly object based on config.
    """

    def __init__(self):
        self.assembly_map = {
            "AggregatorAssembly": AggregatorAssembly,
            "CnnDenseAssembly": CnnDenseAssembly,
            "RnnDenseAssembly": RnnDenseAssembly,
            "CnnRnnDenseAssembly": CnnRnnDenseAssembly,
            "DenseAssembly": DenseAssembly,
            "ResNetAssembly": ResNetAssembly,
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
            "DenseAssembly": self._dense_assembly_init,
            "ResNetAssembly": self._resnet_assembly_init,
            "AggregatorAssembly": self._aggregator_assembly_init,
        }

    def get_pretrained_torch_model(self):
        raise NotImplementedError("Implement this method")

    def get_model_from_pretrained(self):
        raise NotImplementedError("Implement this method")

    def get_model_assembly(
        self, assembly_config: dict
    ) -> Union["Assembly", AggregatorAssembly]:
        assembly_class_name = assembly_config.get("class_name", "")
        if not assembly_class_name:
            raise KeyError("assembly_config doesn't contain `class_name` key")

        return self.initializer_map[assembly_class_name](
            assembly_config, assembly_class_name
        )

    def _aggregator_assembly_init(
        self, assembly_config: dict, assembly_class_name: str
    ):
        classifier_cfg = assembly_config.get("classifier", {})
        embedding_model_ckpt_filename = assembly_config.get(
            "embedding_model_ckpt_filename"
        )
        embedding_model_config = assembly_config.get("embedding_model_config")
        map_location = assembly_config.get("map_location")
        aggregator_type = assembly_config.get("aggregator_type")
        trainable_aggregator = assembly_config.get("trainable_aggregator")
        if not embedding_model_ckpt_filename:
            raise KeyError(
                "assembly_config doesn't contain `embedding_model_ckpt_filename` key"
            )
        if not embedding_model_config:
            raise KeyError(
                "assembly_config doesn't contain `embedding_model_config` key"
            )
        if not map_location:
            raise KeyError("assembly_config doesn't contain `map_location` key")
        if not aggregator_type:
            raise KeyError("assembly_config doesn't contain `aggregator_type` key")
        if trainable_aggregator is None:
            raise KeyError("assembly_config doesn't contain `trainable_aggregator` key")

        embedding_model_assembly = self.get_model_assembly(
            assembly_config=embedding_model_config
        )
        ckpt = torch.load(
            TRAINED_MODELS_DIR / embedding_model_ckpt_filename,
            map_location=map_location,
        )
        state_dict = ckpt["state_dict"]

        # TrainerModule: self.model = model
        state_dict = {
            k.removeprefix("model."): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        embedding_model_assembly.load_state_dict(state_dict=state_dict)

        aggregator_assembly = self.assembly_map[assembly_class_name](
            embedding_model=embedding_model_assembly,
            aggregator_type=aggregator_type,
            trainable_aggregator=trainable_aggregator,
        )
        aggregator_assembly.init_classifier(**classifier_cfg)
        return aggregator_assembly

    def _resnet_assembly_init(self, assembly_config: dict, assembly_class_name: str):
        backbone_name = assembly_config.get("backbone_name", False)
        weights = assembly_config.get("weights", False)
        freeze_backbone = assembly_config.get("freeze_backbone", None)
        sequence_encoder_cfg = assembly_config.get("sequence_encoder", {})
        classifier_cfg = assembly_config.get("classifier", {})
        if not backbone_name:
            raise KeyError("assembly_config doesn't contain `backbone_name` key")
        if weights is False:
            raise KeyError("assembly_config doesn't contain `weights` key")
        if freeze_backbone is None:
            raise KeyError("assembly_config doesn't contain `freeze_backbone` key")
        if not sequence_encoder_cfg:
            raise KeyError("assembly_config doesn't contain `sequence_encoder` key")
        if not classifier_cfg:
            raise KeyError("assembly_config doesn't contain `classifier` key")
        assembly = self.assembly_map[assembly_class_name](
            backbone_name=backbone_name,
            weights=weights,
            freeze_backbone=freeze_backbone,
        )
        assembly.init_seq_encoder(**sequence_encoder_cfg)
        assembly.init_classifier(**classifier_cfg)
        return assembly

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

    def _dense_assembly_init(
        self, assembly_config: dict, assembly_class_name: str
    ) -> "Assembly":
        feature_extractor_cfg = assembly_config.get("feature_extractor", {})
        classifier_cfg = assembly_config.get("classifier", {})
        if not feature_extractor_cfg:
            raise KeyError("assembly_config doesn't contain `classifier` key")
        if not classifier_cfg:
            raise KeyError("assembly_config doesn't contain `classifier` key")
        assembly = self.assembly_map[assembly_class_name]()
        assembly.init_feature_extractor(**feature_extractor_cfg)
        assembly.init_classifier(**classifier_cfg)
        return assembly
