from dataclasses import dataclass


def _check_bounds(r: list[int, int] | list[float, float], name: str):
    if len(r) != 2 or r[0] > r[1]:
        raise ValueError(f"'{name}' failed post init validation")


@dataclass(frozen=True)
class RnnDenseBounds:
    n_seq_encoder_layers: list[int, int]
    dropout: list[float, float]

    def __post_init__(self):
        _check_bounds(self.n_seq_encoder_layers, "n_seq_encoder_layers")
        _check_bounds(self.dropout, "dropout")


@dataclass(frozen=True)
class CnnDenseBounds:
    n_blocks: list[int, int]
    n_layers_per_block: list[int, int]
    n_filters_per_block: list[int, int]
    n_filters_per_skip: list[int, int]

    n_seq_encoder_layers: list[int, int]
    n_units_per_seq_encoder_layer: list[int, int]

    def __post_init__(self):
        _check_bounds(self.n_blocks, "n_blocks")
        _check_bounds(self.n_layers_per_block, "n_layers_per_block")
        _check_bounds(self.n_filters_per_block, "n_filters_per_block")
        _check_bounds(self.n_filters_per_skip, "n_filters_per_skip")
        _check_bounds(
            self.n_units_per_seq_encoder_layer, "n_units_per_seq_encoder_layer"
        )


@dataclass(frozen=True)
class CnnRnnDenseBounds:
    n_blocks: list[int, int]
    n_layers_per_block: list[int, int]
    n_filters_per_block: list[int, int]
    n_filters_per_skip: list[int, int]

    n_seq_encoder_layers: list[int, int]
    dropout: list[float, float]

    def __post_init__(self):
        _check_bounds(self.n_blocks, "n_blocks")
        _check_bounds(self.n_layers_per_block, "n_layers_per_block")
        _check_bounds(self.n_filters_per_block, "n_filters_per_block")
        _check_bounds(self.n_filters_per_skip, "n_filters_per_skip")
        _check_bounds(self.n_seq_encoder_layers, "n_seq_encoder_layers")
        _check_bounds(self.dropout, "dropout")


@dataclass(frozen=True)
class DenseBounds:
    n_feature_extractor_layers: list[int, int]
    n_units_per_feature_extractor_layer: list[int, int]

    def __post_init__(self):
        _check_bounds(self.n_feature_extractor_layers, "n_feature_extractor_layers")
        _check_bounds(
            self.n_units_per_feature_extractor_layer,
            "n_units_per_feature_extractor_layer",
        )


@dataclass(frozen=True)
class Blocks:
    """
    Full list of blocks fo CnnDenseAssembly:
        [
            "Conv1DBlockWithDilationWithSkip",
            "Conv1DBlockWithDilationNoSkip",
            "Conv1DBlockNoDilationWithSkip",
            "Conv1DBlockNoDilationNoSkip",
            "Conv2DBlockWithSkip",
            "Conv2DBlockNoSkip",
        ]

    Full list of blocks for CnnRnnDenseAssembly:
        [
            "Conv1DBlockWithDilationWithSkip",
            "Conv1DBlockWithDilationNoSkip",
            "Conv1DBlockNoDilationWithSkip",
            "Conv1DBlockNoDilationNoSkip",
        ]

    If optuna run is not searching through the given architecture put None to its blocks.
    """

    cnn_dense_assembly_blocks: list[str] | None
    cnn_rnn_dense_assembly_blocks: list[str] | None

    def __post_init__(self):
        if self.cnn_rnn_dense_assembly_blocks is not None and not len(
            self.cnn_rnn_dense_assembly_blocks
        ):
            raise ValueError(
                "cnn_rnn__dense_assembly_blocks has to contain at least one block."
            )
        if self.cnn_dense_assembly_blocks is not None and not len(
            self.cnn_dense_assembly_blocks
        ):
            raise ValueError(
                "cnn_rnn__dense_assembly_blocks has to contain at least one block."
            )
