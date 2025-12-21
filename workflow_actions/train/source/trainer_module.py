import lightning as L
import torch.nn as nn
import torch
from workflow_actions.dataset_preprocessor.run import (
    dp,
)
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OptimizationParamsConfig:
    lr_schedule: Literal["warmup_cosine"] | None = None
    lr_schedule_params: dict | None = None
    optimizer: Literal["AdamW", "Adam"] = "Adam"
    optimizer_params: dict | None = None


class TrainerModule(L.LightningModule):
    def __init__(self, model, optimization: dict, do_pre_epoch_hook=False):
        super().__init__()
        self.model = model
        self.optimization_params = OptimizationParamsConfig(**optimization)
        self.criterion = nn.BCELoss()
        self.validation_predictions = []
        self.validation_targets = []

        self.do_pre_epoch_hook = do_pre_epoch_hook

        self._optimizers_map = {
            "AdamW": torch.optim.AdamW,
            "Adam": torch.optim.Adam,
        }

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        cfg = self.optimization_params
        optimizer_cls = self._optimizers_map[cfg.optimizer]

        if cfg.optimizer_params is None:
            optimizer = optimizer_cls(self.parameters())
        else:
            optimizer = optimizer_cls(self.parameters(), **cfg.optimizer_params)

        if cfg.lr_schedule is None:
            return optimizer

        if cfg.lr_schedule == "warmup_cosine":
            params = cfg.lr_schedule_params or {}
            start_factor = params.get("start_factor", 0.1)
            warmup_iters = params.get("warmup_iters", 5)
            T_max = params.get("T_max", 45)
            eta_min = params.get("eta_min", 0.0)

            warmup = LinearLR(
                optimizer,
                start_factor=start_factor,
                total_iters=warmup_iters,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_iters],
            )
            return [optimizer], [scheduler]

        return optimizer

    def on_train_epoch_end(self) -> None:
        if self.do_pre_epoch_hook:
            dp.prepare_model_ready_data()
