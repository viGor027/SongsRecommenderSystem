import lightning as L
import torch.nn as nn
import torch
from workflow_actions.dataset_preprocessor.dataset_preprocessor import DatasetPreprocessor
from workflow_actions.paths import DATASET_PREPROCESSOR_CONFIG_PATH
from workflow_actions.json_handlers import read_json_to_dict


class TrainerModule(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.validation_predictions = []
        self.validation_targets = []

        prepare_dataset_cfg = read_json_to_dict(DATASET_PREPROCESSOR_CONFIG_PATH)
        self.dp = DatasetPreprocessor(**prepare_dataset_cfg)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_start(self) -> None:
        self.dp.pre_epoch_augment_hook()
