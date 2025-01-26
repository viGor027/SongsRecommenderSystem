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
from cloud.cloud_utils import save_dict_to_gcs_as_json, load_tensor_from_gcs, get_ready_model_from_gcs_checkpoint, \
    read_json_from_gcs_to_dict
from song_pipeline.constants import DATA_DIR

model_id_str = 'model_'
data_version_str = 'data_4'
bucket_name = 'grid_3'

X_train = load_tensor_from_gcs(bucket_name="data_versions", folder_name=data_version_str, tensor_name="X_1_train")
Y_train = load_tensor_from_gcs(bucket_name="data_versions", folder_name=data_version_str, tensor_name="Y_1_train")

X_valid = load_tensor_from_gcs(bucket_name="data_versions", folder_name=data_version_str, tensor_name="X_1_valid")
Y_valid = load_tensor_from_gcs(bucket_name="data_versions", folder_name=data_version_str, tensor_name="Y_1_valid")

for i in [2, 3]:
    X_train_temp = load_tensor_from_gcs(bucket_name="data_versions", folder_name=data_version_str,
                                        tensor_name=f"X_{i}_train")
    Y_train_temp = load_tensor_from_gcs(bucket_name="data_versions", folder_name=data_version_str,
                                        tensor_name=f"Y_{i}_train")

    X_valid_temp = load_tensor_from_gcs(bucket_name="data_versions", folder_name=data_version_str,
                                        tensor_name=f"X_{i}_valid")
    Y_valid_temp = load_tensor_from_gcs(bucket_name="data_versions", folder_name=data_version_str,
                                        tensor_name=f"Y_{i}_valid")

    X_train = torch.concat([X_train, X_train_temp], dim=0)
    Y_train = torch.concat([Y_train, Y_train_temp], dim=0)

    X_valid = torch.concat([X_valid, X_valid_temp], dim=0)
    Y_valid = torch.concat([Y_valid, Y_valid_temp], dim=0)

# print(X_train.shape)
X_train, Y_train = X_train.float(), Y_train.float()

# print(X_valid.shape)
X_valid, Y_valid = X_valid.float(), Y_valid.float()

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_valid, Y_valid)
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)

model_cfg = read_json_from_gcs_to_dict(bucket_name='fine_tuned', folder_name='model_1', file_name='cfg')
model_cfg["temporal_compressor"]["ConvCls"] = Conv1DBlockNoDilationNoSkip

model = CnnRnnDenseAssembly()
model.init_conv(**model_cfg["temporal_compressor"])
model.init_seq_encoder(**model_cfg["sequence_encoder"])
model.init_classifier(**model_cfg["classifier"])

i = 3
gcs_checkpoint_callback = GCSModelCheckpoint(
    bucket_name=bucket_name,
    folder_name=model_id_str + str(i),
    checkpoint_name="model"
)
early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=3,
    verbose=True
)
module = TrainerModule(model, learning_rate=1e-3)
trainer = L.Trainer(
    max_epochs=150,
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
    bucket_name=bucket_name,
    folder_name=model_id_str + str(i),
    file_name="cfg"
)