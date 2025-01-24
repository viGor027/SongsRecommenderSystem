import os.path
import torch
from torch.utils.data import TensorDataset, DataLoader
from model_components.temporal_compressor.conv1d_block_with_dilation_with_skip import Conv1DBlockWithDilationWithSkip
from model_components.temporal_compressor.conv1d_block_with_dilation_no_skip import Conv1DBlockWithDilationNoSkip
from model_components.temporal_compressor.conv1d_block_no_dilation_no_skip import Conv1DBlockNoDilationNoSkip
from model_components.temporal_compressor.conv1d_block_no_dilation_with_skip import Conv1DBlockNoDilationWithSkip
from prototyping.assemblies.cnn_dense_assembly import CnnDenseAssembly
from prototyping.assemblies.cnn_rnn_dense_assembly import CnnRnnDenseAssembly
from prototyping.assemblies.rnn_dense_assembly import RnnDenseAssembly
from prototyping.trainer_module import TrainerModule
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from prototyping.checkpoint_callback import GCSModelCheckpoint
from cloud.cloud_utils import save_dict_to_gcs_as_json, load_tensor_from_gcs, get_ready_model_from_gcs_checkpoint
from song_pipeline.constants import DATA_DIR

model_id_str = 'model_1'
data_version_str = 'data_any'

early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=3,
    verbose=True
)

gcs_checkpoint_callback = GCSModelCheckpoint(
    bucket_name="models_training_ckpt",
    folder_name="RnnDenseTest",
    checkpoint_name="model"
)

n_blocks = 4
no_skip_filters = [0 for _ in range(n_blocks)]
n_filters_per_skip = [16 + 2 * i for i in range(n_blocks)]

model = get_ready_model_from_gcs_checkpoint(
    bucket_name="models_training_ckpt",
    folder_name="RnnDenseTest",
    checkpoint_name='model',
    cfg_file_name="cfg")

# model = RnnDenseAssembly()

# CnnRnnDenseAssembly
# model.init_seq_encoder(n_input_channels=80,
#                        n_seq_encoder_layers=4,
#                        hidden_size=128,
#                        dropout=0.3,
#                        layer_type='gru')
#
# model.init_classifier(n_classifier_layers=0,
#                       n_units_per_classifier_layer=[],
#                       n_classes=91)

X_train = torch.load(os.path.join(DATA_DIR, "X_1_valid.pt"))[:128]
Y_train = torch.load(os.path.join(DATA_DIR, "Y_1_valid.pt"))[:128]
print(X_train.shape)
X_train, Y_train = X_train.float(), Y_train.float()

X_valid = torch.load(os.path.join(DATA_DIR, "X_2_valid.pt"))[:128]
Y_valid = torch.load(os.path.join(DATA_DIR, "Y_2_valid.pt"))[:128]
print(X_valid.shape)
X_valid, Y_valid = X_valid.float(), Y_valid.float()

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_valid, Y_valid)
batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

module = TrainerModule(model, learning_rate=1e-4)
trainer = L.Trainer(
    max_epochs=1,
    callbacks=[early_stopping_callback, gcs_checkpoint_callback],
    accelerator="auto",
    devices="auto"
)
trainer.fit(module, train_loader, val_loader)

model_dct = model.get_instance_config()
model_dct['best_loss'] = gcs_checkpoint_callback.retrieve_best_loss().item()
model_dct['data_version'] = data_version_str
print(model_dct)
save_dict_to_gcs_as_json(
    model_dct,
    bucket_name="models_training_ckpt",
    folder_name="RnnDenseTest",
    file_name="cfg"
)
