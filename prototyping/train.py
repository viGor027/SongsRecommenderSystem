import os.path
import torch
from torch.utils.data import TensorDataset, DataLoader
from model_components.temporal_compressor.conv1d_block_with_dilation_with_skip import Conv1DBlockWithDilationWithSkip
from model_components.temporal_compressor.conv1d_block_with_dilation_no_skip import Conv1DBlockWithDilationNoSkip
from prototyping.assemblies.cnn_dense_assembly import CnnDenseAssembly
from prototyping.assemblies.cnn_rnn_dense_assembly import CnnRnnDenseAssembly
from prototyping.trainer_module import TrainerModule
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from prototyping.checkpoint_callback import GCSModelCheckpoint
from cloud.cloud_utils import save_dict_to_gcs_as_json, load_tensor_from_gcs
from song_pipeline.constants import DATA_DIR

early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=5,
    verbose=True
)

gcs_checkpoint_callback = GCSModelCheckpoint(
    bucket_name="models_training_ckpt",
    folder_name="test_cloud",
    checkpoint_name="test_cloud_ckpt"
)

n_blocks = 4
no_skip_filters_per_skip = [0 for _ in range(n_blocks)]
n_filters_per_skip = [16 + 2*i for i in range(n_blocks)]

model = CnnDenseAssembly()
model.init_conv(ConvCls=Conv1DBlockWithDilationWithSkip,
                n_blocks=n_blocks,
                n_layers_per_block=[1 for _ in range(n_blocks)],
                n_filters_per_block=[64+8*i for i in range(n_blocks)],
                n_filters_per_skip=n_filters_per_skip,
                reduction_strat='conv',
                input_len=431, n_input_channels=80)
# CnnDenseAssembly
# units_per_layer = [312, 256, 224, 192, 160, 128]

model.init_seq_encoder(
    n_seq_encoder_layers=1,
    n_units_per_seq_encoder_layer=[312],
    n_embedding_dims=256
)
model.init_classifier(
    n_classifier_layers=4,
    n_units_per_classifier_layer=[224, 192, 160, 128],
    n_classes=91
)


# CnnRnnDenseAssembly
# model.init_seq_encoder(n_seq_encoder_layers=4,
#                        hidden_size=512,
#                        dropout=0.4,
#                        layer_type='gru')
#
# model.init_classifier(n_classifier_layers=3,
#                       n_units_per_classifier_layer=[192, 160, 128],
#                       n_classes=91)

# model_dct = model.get_instance_config()
# print(model_dct)
# save_dict_to_gcs_as_json(
#     model_dct,
#     bucket_name="models_training_ckpt",
#     folder_name="test_cloud",
#     file_name="model_spec_test_cloud"
# )

# X = load_tensor_from_gcs(bucket_name="data_versions", folder_name="data_1", tensor_name='X')
# Y = load_tensor_from_gcs(bucket_name="data_versions", folder_name="data_1", tensor_name='Y')

X = torch.load(os.path.join(DATA_DIR, 'X.pt'))
Y = torch.load(os.path.join(DATA_DIR, 'Y.pt'))

X, Y = X.float(), Y.float()

X_train, Y_train, X_valid, Y_valid = X[:20_000], Y[:20_000], X[20_000:], Y[20_000:]

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_valid, Y_valid)
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

module = TrainerModule(model, learning_rate=1e-3)
trainer = L.Trainer(
    max_epochs=100,
    callbacks=[early_stopping_callback],
    accelerator="auto",
    devices="auto"
)
trainer.fit(module, train_loader, val_loader)
