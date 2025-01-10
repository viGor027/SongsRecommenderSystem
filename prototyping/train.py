import torch
from torch.utils.data import TensorDataset, DataLoader
from model_components.temporal_compressor.conv1d_block_no_dilation_no_skip import Conv1DBlockNoDilationNoSkip
from model_components.temporal_compressor.conv1d_block_with_dilation_with_skip import Conv1DBlockWithDilationWithSkip
from prototyping.temporal_compressor_assembly import TemporalCompressorAssembly
from prototyping.trainer_module import TrainerModule
from song_pipeline.constants import DATA_DIR
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from prototyping.checkpoint_callback import GCSModelCheckpoint
from cloud.cloud_utils import load_checkpoint_from_gcs
import os


early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=8,
    verbose=True
)

gcs_checkpoint_callback = GCSModelCheckpoint(
    bucket_name="models_training_ckpt",
    folder_name="test",
    checkpoint_name="test_ckpt"
)

model = TemporalCompressorAssembly(ConvCls=Conv1DBlockWithDilationWithSkip,
                                   n_blocks=4,
                                   n_layers_per_block=[1 for _ in range(7)],
                                   n_filters_per_block=[64+8*i for i in range(7)],
                                   n_filters_per_skip=[16 + 2*i for i in range(7)],
                                   input_len=431, n_input_channels=80,
                                   n_classes=91
                                   )

print(model.save_instance())

X, Y = torch.load(os.path.join(DATA_DIR, 'X.pt')), torch.load(os.path.join(DATA_DIR, 'Y.pt'))

X, Y = X.float(), Y.float()
print(X.shape, Y.shape, X.dtype, Y.dtype)
X_train, Y_train, X_valid, Y_valid = X[:20_000], Y[:20_000], X[20_000:], Y[20_000:]
print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_valid, Y_valid)
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

module = TrainerModule(model, learning_rate=1e-3)
trainer = L.Trainer(max_epochs=20, callbacks=[early_stopping_callback, gcs_checkpoint_callback])
trainer.fit(module, train_loader, val_loader)
