from workflow_actions.paths import MODEL_READY_DATA_DIR
from workflow_actions.train.source import ModelInitializer
import torch

dense_cfg = {
    "class_name": "DenseAssembly",
    "classifier": {
        "n_classes": 110,
        "n_classifier_layers": 1,
        "classifier_activation": None,
        "n_units_per_classifier_layer": [],
    },
    "feature_extractor": {
        "n_embedding_dims": 384,
        "n_input_channels": 17280,
        "n_feature_extractor_layers": 6,
        "feature_extractor_activation": "relu",
        "n_units_per_feature_extractor_layer": [768, 768, 768, 640, 672],
    },
}

aggregator_cfg = {
    "class_name": "AggregatorAssembly",
    "aggregator_type": "lstm",
    "trainable_aggregator": False,
    "embedding_model_ckpt_filename": "resnet101_IMAGENET1K_V1_raw_aug_spec_aug-epoch=26-val_loss=0.0218_64_batch.ckpt",
    "embedding_model_config": {
        "backbone_name": "resnet101",
        "freeze_backbone": False,
        "weights": "IMAGENET1K_V1",
        "class_name": "ResNetAssembly",
        "sequence_encoder": {
            "n_embedding_dims": 384,
            "n_seq_encoder_layers": 1,
            "seq_encoder_activation": None,
            "n_units_per_seq_encoder_layer": [],
        },
        "classifier": {
            "n_classes": 110,
            "sigmoid_output": True,
            "n_classifier_layers": 1,
            "classifier_activation": None,
            "n_units_per_classifier_layer": [],
        },
    },
    "classifier": {"n_classes": 110},
    "map_location": "cpu",
}

sample_1 = torch.load(MODEL_READY_DATA_DIR / "train" / "X_0.pt")
sample_2 = torch.load(MODEL_READY_DATA_DIR / "train" / "X_1.pt")

aggregator_input = torch.cat([sample_1, sample_2])
dense_input = sample_1

mi = ModelInitializer()
aggregator_model = mi.get_model_assembly(assembly_config=aggregator_cfg).eval()
dense_model = mi.get_model_assembly(assembly_config=dense_cfg).eval()

emb_dense = dense_model.make_embeddings(dense_input)
prediction_dense = dense_model.forward(dense_input)

emb_agg = aggregator_model.aggregate_fragments(aggregator_input)
prediction_agg = aggregator_model.forward(aggregator_input)

print(
    f"dense model input sample size: {dense_input.size()}\n"
    + f"aggregator model input sample size: {aggregator_input.size()}\n"
    + f"emb_dense size: {emb_dense.size()}\n"
    + f"prediction_dense size: {prediction_dense.size()}\n"
    + f"emb_agg size: {emb_agg.size()}\n"
    + f"prediction_agg size: {prediction_agg.size()}"
)
