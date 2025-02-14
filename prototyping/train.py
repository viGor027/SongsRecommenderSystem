from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from model_components.temporal_compressor.conv1d_block_with_dilation_with_skip import Conv1DBlockWithDilationWithSkip
from model_components.temporal_compressor.conv1d_block_with_dilation_no_skip import Conv1DBlockWithDilationNoSkip
from model_components.temporal_compressor.conv1d_block_no_dilation_no_skip import Conv1DBlockNoDilationNoSkip
from model_components.temporal_compressor.conv1d_block_no_dilation_with_skip import Conv1DBlockNoDilationWithSkip

from prototyping.assemblies.cnn_dense_assembly import CnnDenseAssembly
from prototyping.assemblies.cnn_rnn_dense_assembly import CnnRnnDenseAssembly
from prototyping.assemblies.rnn_dense_assembly import RnnDenseAssembly

from prototyping.checkpoint_callback import GCSModelCheckpoint
from prototyping.trainer_module import TrainerModule
from prototyping.dataset import BatchedDataset

from song_pipeline.utils import read_json_to_dict
from song_pipeline.constants import PROJECT_FOLDER_DIR
import os

model_id_str = 'model_'
data_version_str = 'data_4'
bucket_name = 'grid_3'


train_dataset = BatchedDataset('train')

val_dataset = BatchedDataset('valid')

train_loader = DataLoader(train_dataset, batch_size=None,
                          collate_fn=BatchedDataset.collate,
                          shuffle=True, num_workers=0)

val_loader = DataLoader(val_dataset, batch_size=None,
                        collate_fn=BatchedDataset.collate,
                        num_workers=0)

model_cfg = read_json_to_dict(
    os.path.join(PROJECT_FOLDER_DIR, 'downloads', 'cloud_save', 'model', 'cfg.json')

)

model_cfg["temporal_compressor"]["ConvCls"] = Conv1DBlockNoDilationNoSkip

model = CnnRnnDenseAssembly()
model.init_conv(**model_cfg["temporal_compressor"])
model.init_seq_encoder(**model_cfg["sequence_encoder"])
model.init_classifier(**model_cfg["classifier"])

gcs_checkpoint_callback = GCSModelCheckpoint(
    bucket_name="",
    folder_name="",
    checkpoint_name=""
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=3,
    verbose=True
)

module = TrainerModule(model, learning_rate=1e-3)
trainer = L.Trainer(
    max_epochs=1,
    callbacks=[early_stopping_callback],
    accelerator="gpu",
    devices="auto",
    precision="32",
)

trainer.fit(module, train_loader, val_loader)

model_dct = model.get_instance_config()
# model_dct['best_loss'] = gcs_checkpoint_callback.retrieve_best_loss().item()
model_dct['data_version'] = data_version_str

print(model_dct)
