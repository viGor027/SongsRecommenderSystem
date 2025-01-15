import os
import torch
import lightning as L
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model_components.temporal_compressor.conv1d_block_with_dilation_with_skip import Conv1DBlockWithDilationWithSkip
from model_components.classifier.base_classifier import BaseClassifier
from song_pipeline.constants import DATA_DIR
from prototyping.assemblies.cnn_dense_assembly import TemporalCompressorAssembly


class TrainerModule(L.LightningModule):
    #def __init__(self, c1, c2, c3, c4, c5, c6, classifier, learning_rate=1e-3):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        # self.c1 = c1
        # self.c2 = c2
        # self.c3 = c3
        # self.c4 = c4
        # self.c5 = c5
        # self.c6 = c6
        # self.classifier = classifier
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        # self.criterion = nn.L1Loss()  # MAE
        self.validation_predictions = []
        self.validation_targets = []

    def forward(self, x):
        # x = self.c1(x)
        # x = self.c2(x)
        # x = self.c3(x)
        # x = self.c4(x)
        # x = self.c5(x)
        # x = self.c6(x)
        # x = x.reshape(x.size(0), -1)
        # x = self.classifier(x)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        # print(f"BATCH_{batch_idx}")
        x, y = batch
        y_pred = self(x)
        # print("Y_PRED:")
        # print(y_pred)
        # print("Y_TRUE")
        # print(y)
        # print()
        loss = self.criterion(y_pred, y)
        print(f"LOSS: {loss.item()}")
        self.log("train_loss", loss, prog_bar=True)
        # Print gradients after the first batch
        # if batch_idx == 0:
        #     self.zero_grad()  # Ensure no accumulated gradients
        #     loss.backward()  # Manually compute gradients for inspection
        #     for name, param in self.model.named_parameters():
        #         if param.requires_grad:
        #             print(f"Gradients for {name}: {param.grad}")
        #     self.zero_grad()  # Clear gradients to avoid interference with Lightning's backward pass

        # for el in self.parameters():
        #     print(f"PARAM_batch_idx_{batch_idx}")
        #     print(el.grad)
        return loss

    # def on_before_optimizer_step(self, optimizer):
    #     print("PRINTING GRADIENTS")
    #     for el in self.parameters():
    #         print(el.grad)

    def validation_step(self, batch, batch_idx):
        # print("WYKONUJE SIE VALID_STEP")
        x, y = batch
        y_pred = self(x)
        # print("Y_PRED")
        # print(y_pred)
        # print("Y")
        # print(y)
        loss = self.criterion(y_pred, y)
        self.validation_predictions.append(y_pred.detach().cpu())
        self.validation_targets.append(y.detach().cpu())
        self.log("val_loss", loss, prog_bar=True)
        # print("WYKONALO SIE VALID_STEP")

    # def on_validation_epoch_end(self):
    #     # Combine all predictions and targets
    #     preds = torch.cat(self.validation_predictions)
    #     targets = torch.cat(self.validation_targets)
    #     self.validation_predictions.clear()
    #     self.validation_targets.clear()
    #
    #     # Print F1 score and loss
    #     val_loss = self.trainer.callback_metrics["val_loss"].item()
    #     #print(f"Validation Loss: {val_loss:.4f}, F1 Score: {f1:.4f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# min_dataset_dir = os.path.join(DATA_DIR, 'min_dataset')
# X, Y = torch.load(os.path.join(min_dataset_dir, 'X_min.pt')), torch.load(os.path.join(min_dataset_dir, 'Y_min.pt'))

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
n_filters = 64
n_skip = 32

# outputs sequences of len 6
# Initialize Model
n_layers = 1
n_input_features = 6 * (n_filters+n_skip)
units_per_layer = [4 * 32]
n_classes = 91

classifier = BaseClassifier(
    n_layers=n_layers,
    n_input_features=n_input_features,
    units_per_layer=units_per_layer,
    n_classes=n_classes
)

#model = Assembly(ConvCls=Conv1DBlockWithDilationWithSkip, n_blocks=7, n_filters=n_filters, n_skip=n_skip, input_len=431, n_input_channels=80)

n_blocks = 7
n_layers_per_block = [1 for _ in range(n_blocks)]
n_filters_per_block = [n_filters + 8*i for i in range(n_blocks)]
n_filters_per_skip = [n_skip + 2*i for i in range(n_blocks)]
input_len = 431
n_input_channels = 80
model = TemporalCompressorAssembly(
    ConvCls=Conv1DBlockWithDilationWithSkip,
    n_blocks=n_blocks,
    n_layers_per_block=n_layers_per_block,
    n_filters_per_block=n_filters_per_block,
    n_filters_per_skip=n_filters_per_skip,
    input_len=input_len,
    n_input_channels=n_input_channels,
    n_classes=n_classes
)
# Training
binary_classifier = TrainerModule(model, learning_rate=1e-3)
trainer = L.Trainer(max_epochs=1, log_every_n_steps=10)
trainer.fit(binary_classifier, train_loader, val_loader)


from cloud.tut.cloud_utils import save_model_to_srs_models, load_model_from_srs_models

save_model_to_srs_models(model, 'srs_models', 'test', 'model_1')
model = load_model_from_srs_models('srs_models', 'test', 'model_1')

print("#################################\n NEW RUN\n#################################\n")

binary_classifier = TrainerModule(model, learning_rate=1e-3)
trainer = L.Trainer(max_epochs=1, log_every_n_steps=10)
trainer.fit(binary_classifier, train_loader, val_loader)