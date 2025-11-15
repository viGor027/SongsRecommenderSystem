import optuna
from workflow_actions.train.source.optuna_config_builder_config_schemas import (
    RnnDenseBounds,
    CnnDenseBounds,
    CnnRnnDenseBounds,
    DenseBounds,
    Blocks,
)


class OptunaAssemblyConfigBuilder:
    def __init__(
        self,
        rnn_dense_bounds: dict | None,
        cnn_dense_bounds: dict | None,
        cnn_rnn_dense_bounds: dict | None,
        dense_bounds: dict | None,
        blocks: dict,
        assemblies: list[str],
        input_len: int,
        n_input_channels: int,
        n_embedding_dims: int,
        n_classes: int,
        optuna_step: int,
        trial: optuna.Trial | None = None,
    ):
        """
        Notes:
            - assemblies is a list of assemblies assembly for trial will be chosen from
            - n_classes depends on the data
            - n_input_channels is used for RnnDenseAssembly and CnnDenseAssembly when Conv1D blocks are used
            - input_len is used if Conv1D blocks are used
            - n_embedding_dims is used as hidden_size for RnnDenseAssembly and CnnRnnDenseAssembly
            because that corresponds to the number of embedding dims
        """
        self.trial = trial

        # Assemblies bounds
        self.rnn_dense_bounds = (
            RnnDenseBounds(**rnn_dense_bounds) if rnn_dense_bounds is not None else None
        )
        self.cnn_dense_bounds = (
            CnnDenseBounds(**cnn_dense_bounds) if cnn_dense_bounds is not None else None
        )
        self.cnn_rnn_dense_bounds = (
            CnnRnnDenseBounds(**cnn_rnn_dense_bounds)
            if cnn_rnn_dense_bounds is not None
            else None
        )
        self.dense_bounds = (
            DenseBounds(**dense_bounds) if dense_bounds is not None else None
        )

        self.blocks = Blocks(**blocks)

        self.input_len = input_len
        self.n_input_channels = n_input_channels
        self.n_embedding_dims = n_embedding_dims
        self.n_classes = n_classes

        self.optuna_step = optuna_step

        self.available_assemblies = assemblies

        self.CONV2D_BLOCK_NAMES = ["Conv2DBlockWithSkip", "Conv2DBlockNoSkip"]

        self.CONV_BLOCK_WITHOUT_SKIP_NAMES = [
            "Conv1DBlockWithDilationNoSkip",
            "Conv1DBlockNoDilationNoSkip",
            "Conv2DBlockNoSkip",
        ]

        self.ACTIVATIONS = ["relu", "hardswish"]

        self.builders_map = {
            "RnnDenseAssembly": self._build_rnn_dense_cfg,
            "CnnDenseAssembly": self._build_cnn_dense_cfg,
            "CnnRnnDenseAssembly": self._build_cnn_rnn_dense_cfg,
            "DenseAssembly": self._build_dense_cfg,
        }

    def build_assembly_config(self) -> tuple[dict, str]:
        """returns assembly config and assembly_type"""
        if self.trial is None:
            raise ValueError("trial can't be None.")
        assembly_type = self.trial.suggest_categorical(
            "assembly_type",
            self.available_assemblies,
            # ["CnnDenseAssembly", "RnnDenseAssembly", "CnnRnnDenseAssembly", "DenseAssembly"],
        )
        return self.builders_map[assembly_type](
            assembly_name=assembly_type
        ), assembly_type

    def set_trial(self, trial: optuna.Trial):
        self.trial = trial

    def _build_rnn_dense_cfg(self, assembly_name: str) -> dict:
        return {
            "class_name": "RnnDenseAssembly",
            "sequence_encoder": {
                "n_input_channels": self.n_input_channels,
                "n_seq_encoder_layers": self.trial.suggest_int(
                    f"{assembly_name}/n_seq_encoder_layers",
                    *self.rnn_dense_bounds.n_seq_encoder_layers,
                ),
                "hidden_size": self.n_embedding_dims,
                "dropout": self.trial.suggest_float(
                    f"{assembly_name}/dropout", *self.rnn_dense_bounds.dropout
                ),
                "layer_type": self.trial.suggest_categorical(
                    f"{assembly_name}/layer_type", ["gru", "lstm"]
                ),
            },
            "classifier": self._get_classifier_config(),
        }

    def _build_cnn_dense_cfg(self, assembly_name: str) -> dict:
        conv_cls = self.trial.suggest_categorical(
            f"{assembly_name}/ConvCls",
            self.blocks.cnn_dense_assembly_blocks,
        )
        n_blocks = self.trial.suggest_int(
            f"{assembly_name}/n_blocks", *self.cnn_dense_bounds.n_blocks
        )
        n_seq_encoder_layers = self.trial.suggest_int(
            f"{assembly_name}/n_seq_encoder_layers",
            *self.cnn_dense_bounds.n_seq_encoder_layers,
        )
        return {
            "class_name": "CnnDenseAssembly",
            "temporal_compressor": {
                "ConvCls": conv_cls,
                "input_len": None
                if conv_cls in self.CONV2D_BLOCK_NAMES
                else self.input_len,
                "n_input_channels": 1
                if conv_cls in self.CONV2D_BLOCK_NAMES
                else self.n_input_channels,
                "n_blocks": n_blocks,
                "n_layers_per_block": [
                    self.trial.suggest_int(
                        f"{assembly_name}/n_layers_per_block_{i}",
                        *self.cnn_dense_bounds.n_layers_per_block,
                    )
                    for i in range(n_blocks)
                ],
                "n_filters_per_block": [
                    self.trial.suggest_int(
                        f"{assembly_name}/n_filters_per_block_{i}",
                        *self.cnn_dense_bounds.n_filters_per_block,
                        step=self.optuna_step,
                    )
                    for i in range(n_blocks)
                ],
                "n_filters_per_skip": (
                    None
                    if conv_cls in self.CONV_BLOCK_WITHOUT_SKIP_NAMES
                    else [
                        self.trial.suggest_int(
                            f"{assembly_name}/n_filters_skip_{i}",
                            *self.cnn_dense_bounds.n_filters_per_skip,
                            step=self.optuna_step,
                        )
                        for i in range(n_blocks)
                    ]
                ),
                "conv_activation": self.trial.suggest_categorical(
                    f"{assembly_name}/conv_activation", self.ACTIVATIONS
                ),
                "reduction_strat": self.trial.suggest_categorical(
                    f"{assembly_name}/reduction_strat", ["conv", "max_pool", "avg_pool"]
                ),
            },
            "sequence_encoder": {
                "n_seq_encoder_layers": n_seq_encoder_layers,
                "n_units_per_seq_encoder_layer": [
                    self.trial.suggest_int(
                        f"{assembly_name}/n_units_per_seq_encoder_layer_{i}",
                        *self.cnn_dense_bounds.n_units_per_seq_encoder_layer,
                        step=self.optuna_step,
                    )
                    for i in range(n_seq_encoder_layers - 1)
                ],
                "n_embedding_dims": self.n_embedding_dims,
            },
            "classifier": self._get_classifier_config(),
        }

    def _build_cnn_rnn_dense_cfg(self, assembly_name: str) -> dict:
        conv_cls = self.trial.suggest_categorical(
            f"{assembly_name}/ConvCls",
            self.blocks.cnn_rnn_dense_assembly_blocks,
        )
        n_blocks = self.trial.suggest_int(
            f"{assembly_name}/n_blocks", *self.cnn_rnn_dense_bounds.n_blocks
        )
        return {
            "class_name": "CnnRnnDenseAssembly",
            "temporal_compressor": {
                "ConvCls": conv_cls,
                "input_len": self.input_len,
                "n_input_channels": self.n_input_channels,
                "n_blocks": n_blocks,
                "n_layers_per_block": [
                    self.trial.suggest_int(
                        f"{assembly_name}/n_layers_per_block_{i}",
                        *self.cnn_rnn_dense_bounds.n_layers_per_block,
                    )
                    for i in range(n_blocks)
                ],
                "n_filters_per_block": [
                    self.trial.suggest_int(
                        f"{assembly_name}/n_filters_per_block_{i}",
                        *self.cnn_rnn_dense_bounds.n_filters_per_block,
                        step=self.optuna_step,
                    )
                    for i in range(n_blocks)
                ],
                "n_filters_per_skip": (
                    None
                    if conv_cls in self.CONV_BLOCK_WITHOUT_SKIP_NAMES
                    else [
                        self.trial.suggest_int(
                            f"{assembly_name}/n_filters_skip_{i}",
                            *self.cnn_rnn_dense_bounds.n_filters_per_skip,
                            step=self.optuna_step,
                        )
                        for i in range(n_blocks)
                    ]
                ),
                "conv_activation": self.trial.suggest_categorical(
                    f"{assembly_name}/conv_activation", self.ACTIVATIONS
                ),
                "reduction_strat": self.trial.suggest_categorical(
                    f"{assembly_name}/reduction_strat", ["conv", "max_pool", "avg_pool"]
                ),
            },
            "sequence_encoder": {
                "n_seq_encoder_layers": self.trial.suggest_int(
                    f"{assembly_name}/n_seq_encoder_layers",
                    *self.cnn_rnn_dense_bounds.n_seq_encoder_layers,
                ),
                "hidden_size": self.n_embedding_dims,
                "dropout": self.trial.suggest_float(
                    f"{assembly_name}/dropout", *self.cnn_rnn_dense_bounds.dropout
                ),
                "layer_type": self.trial.suggest_categorical(
                    f"{assembly_name}/layer_type", ["gru", "lstm"]
                ),
            },
            "classifier": self._get_classifier_config(),
        }

    def _build_dense_cfg(self, assembly_name: str):
        n_feature_extractor_layers = self.trial.suggest_int(
            f"{assembly_name}/n_feature_extractor_layers",
            *self.dense_bounds.n_feature_extractor_layers,
        )
        return {
            "class_name": "DenseAssembly",
            "feature_extractor": {
                "n_input_channels": self.input_len * self.n_input_channels,
                "n_feature_extractor_layers": n_feature_extractor_layers,
                "n_units_per_feature_extractor_layer": [
                    self.trial.suggest_int(
                        f"{assembly_name}/n_units_per_feature_extractor_layer_{i}",
                        *self.dense_bounds.n_units_per_feature_extractor_layer,
                        step=self.optuna_step,
                    )
                    for i in range(n_feature_extractor_layers - 1)
                ],
                "n_embedding_dims": self.n_embedding_dims,
                "feature_extractor_activation": self.trial.suggest_categorical(
                    f"{assembly_name}/feature_extractor_activation", self.ACTIVATIONS
                ),
            },
            "classifier": self._get_classifier_config(),
        }

    def _get_classifier_config(self) -> dict:
        return {
            "n_classifier_layers": 1,
            "n_units_per_classifier_layer": [],
            "classifier_activation": None,
            "n_classes": self.n_classes,
        }
