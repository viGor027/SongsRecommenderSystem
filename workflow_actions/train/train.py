import wandb
import optuna
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
from workflow_actions.train.source import (
    TrainerModule,
    ModelInitializer,
    OptunaAssemblyConfigBuilder,
    FragmentsDataset,
)
from workflow_actions.paths import TRAINED_MODELS_DIR
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from functools import partial
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class RunSingleTrainingConfig:
    architecture: dict = field(default_factory=lambda: {})
    dataloaders: dict = field(
        default_factory=lambda: {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": False,
        }
    )
    hparams: dict = field(
        default_factory=lambda: {
            "learning_rate": 1e-3,
            "epochs": 100,
        }
    )
    early_stopping: dict = field(
        default_factory=lambda: {
            "monitor": "val_loss",
            # never use "/" in monitor value; must be same as value logged by TrainerModule
            "mode": "min",
            "patience": 5,
        }
    )
    accelerator: str = "auto"
    precision: str = "32-true"
    project: str = "default_project"
    run_name: str = "default_run"
    do_pre_epoch_hook: bool = False


@dataclass(frozen=True)
class RunOptunaForAssembliesConfig:
    """
    If the batch_size key is present in the dataloaders dictionary,
    its value is treated as fixed and is not included in the hyperparameter search.

    If the learning_rate key is present in the hparams dictionary,
    its value is treated as fixed and is not included in the hyperparameter search.
    """

    dataloaders: dict = field(
        default_factory=lambda: {
            "num_workers": 4,
            "pin_memory": True,
            "drop_last": False,
        }
    )
    hparams: dict = field(
        default_factory=lambda: {
            "epochs": 100,
        }
    )
    early_stopping: dict = field(
        default_factory=lambda: {
            "monitor": "val_loss",
            "mode": "min",
            "patience": 5,
        }
    )
    accelerator: str = "auto"
    precision: str = "32-true"
    n_trials: int = 5_000
    pruner: str = "median"
    sampler: str = "tpe"
    project: str = "default_project"
    study_name: str = "default_study"


class Train:
    def __init__(
        self,
        run_optuna_for_assemblies_config: dict | None = None,
        run_single_training_config: dict | None = None,
        optuna_assembly_config_builder_params: dict | None = None,
    ):
        Path(TRAINED_MODELS_DIR).mkdir(parents=True, exist_ok=True)
        L.seed_everything(42, workers=True)

        self._optuna_assembly_config_builder = (
            OptunaAssemblyConfigBuilder(**optuna_assembly_config_builder_params)
            if optuna_assembly_config_builder_params is not None
            else None
        )

        self.optuna_config = (
            RunOptunaForAssembliesConfig(**run_optuna_for_assemblies_config)
            if run_optuna_for_assemblies_config is not None
            else None
        )

        self.single_training_config = (
            RunSingleTrainingConfig(**run_single_training_config)
            if run_single_training_config is not None
            else None
        )

        self.model_initializer = ModelInitializer()

        self.PRUNERS_MAP = {
            "median": partial(
                MedianPruner,
                n_startup_trials=int(self.optuna_config.n_trials * 0.15),
                n_warmup_steps=5,
                interval_steps=1,
            ),
        }

        self.SAMPLERS_MAP = {
            "tpe": partial(
                TPESampler,
                n_startup_trials=int(self.optuna_config.n_trials * 0.15),
                multivariate=True,
                group=True,
            ),
        }

    @staticmethod
    def get_dataloaders(
        batch_size: int = 32,
        num_workers: int = 4,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        train_dataset = FragmentsDataset(dataset_type="train")
        valid_dataset = FragmentsDataset(dataset_type="valid")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=FragmentsDataset.collate_concat,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=FragmentsDataset.collate_concat,
        )
        return train_loader, valid_loader

    @staticmethod
    def one_model_with_wandb(
        model,
        log_to_wandb_as_config: dict,
        hparams: dict,
        train_dataloader,
        val_dataloader,
        callbacks: dict,
        project: str,
        run_name: str,
        accelerator: str = "auto",
        precision: str = "32-true",
        do_pre_epoch_hook: bool = False,
    ) -> tuple[str | None, float]:
        """
        Trains a single model and logs to Weights & Biases.
        """
        wandb_logger = WandbLogger(
            project=project,
            name=run_name,
            log_model=False,
        )
        wandb_logger.experiment.config.update(
            log_to_wandb_as_config, allow_val_change=True
        )
        module = TrainerModule(
            model,
            learning_rate=hparams["learning_rate"],
            do_pre_epoch_hook=do_pre_epoch_hook,
        )

        enable_ckpt = "model_checkpoint" in callbacks
        trainer = L.Trainer(
            max_epochs=hparams["epochs"],
            logger=wandb_logger,
            callbacks=list(callbacks.values()),
            deterministic=True,
            accelerator=accelerator,
            precision=precision,
            log_every_n_steps=50,
            enable_checkpointing=enable_ckpt,
        )

        try:
            trainer.fit(module, train_dataloader, val_dataloader)
            best_val = callbacks["early_stopping"].best_score
            best_val = float(best_val.item()) if best_val is not None else float("inf")
            wandb_logger.experiment.summary["best_val_loss"] = best_val
        except optuna.TrialPruned as e:
            wandb_logger.experiment.summary.update(
                {"status": "pruned", "reason": str(e)}
            )
            raise
        except Exception as e:
            wandb_logger.experiment.summary.update(
                {"status": "failed", "error": str(e)}
            )
            raise
        finally:
            wandb.finish()

        best_model_path = (
            callbacks["model_checkpoint"].best_model_path
            if callbacks.get("model_checkpoint", False)
            else None
        )
        return best_model_path, best_val

    def run_optuna_for_assemblies(self):
        if self.optuna_config is None or self._optuna_assembly_config_builder is None:
            raise ValueError(
                (
                    "Running optuna requires initializing Train object with "
                    "run_optuna_for_assemblies_config and optuna_assembly_config_builder_params."
                )
            )
        best_model_path = None
        best_score = float("inf")

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_model_path, best_score

            self._optuna_assembly_config_builder.set_trial(trial)

            assembly_cfg, assembly_type = (
                self._optuna_assembly_config_builder.build_assembly_config()
            )
            model = self.model_initializer.get_model_assembly(assembly_cfg)

            learning_rate = (
                self.optuna_config.hparams["learning_rate"]
                if self.optuna_config.hparams.get("learning_rate", False)
                else trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            )
            batch_size = (
                self.optuna_config.dataloaders["batch_size"]
                if self.optuna_config.dataloaders.get("batch_size", False)
                else trial.suggest_categorical(
                    "batch_size", [i for i in range(32, 256 + 1, 32)]
                )
            )
            hparams = {**self.optuna_config.hparams, "learning_rate": learning_rate}
            dataloaders_cfg = {
                **self.optuna_config.dataloaders,
                "batch_size": batch_size,
            }

            train_dataloader, val_dataloader = Train.get_dataloaders(**dataloaders_cfg)

            log_to_wandb_as_config = self._get_log_to_wandb_as_config(
                log_type="optuna",
                architecture=assembly_cfg,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

            run_name = (
                f"{self.optuna_config.study_name}_{assembly_type}_trial_{trial.number}"
            )

            ckpt_path, best_val = Train.one_model_with_wandb(
                model=model,
                log_to_wandb_as_config=log_to_wandb_as_config,
                hparams=hparams,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                project=self.optuna_config.project,
                run_name=run_name,
                accelerator=self.optuna_config.accelerator,
                precision=self.optuna_config.precision,
                callbacks=Train._get_callbacks(
                    early_stopping_cfg=self.optuna_config.early_stopping,
                    checkpoint_name=run_name,
                    trial=trial,
                ),
                do_pre_epoch_hook=False,
            )

            if best_model_path is None or best_val < best_score:
                best_score = best_val
                best_model_path = ckpt_path
            return best_val

        study = optuna.create_study(
            study_name=self.optuna_config.study_name,
            direction="minimize",
            sampler=self.SAMPLERS_MAP[self.optuna_config.sampler](),
            pruner=self.PRUNERS_MAP[self.optuna_config.pruner](),
        )
        study.optimize(objective, n_trials=self.optuna_config.n_trials)

        best_cfg = study.best_trial.params
        return study, best_cfg, best_model_path

    def run_single_training(self) -> str:
        if self.single_training_config is None:
            raise ValueError(
                "Running single training requires initializing Train object with run_single_training_config."
            )
        model = self.model_initializer.get_model_assembly(
            assembly_config=self.single_training_config.architecture
        )
        log_to_wandb_as_config = self._get_log_to_wandb_as_config(
            log_type="single", architecture=model.get_instance_config()
        )
        train_dataloader, val_dataloader = Train.get_dataloaders(
            **self.single_training_config.dataloaders
        )
        best_model_path, _ = Train.one_model_with_wandb(
            model=model,
            hparams=self.single_training_config.hparams,
            log_to_wandb_as_config=log_to_wandb_as_config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            callbacks=Train._get_callbacks(
                early_stopping_cfg=self.single_training_config.early_stopping,
                checkpoint_name=self.single_training_config.run_name,
                trial=None,
            ),
            project=self.single_training_config.project,
            run_name=self.single_training_config.run_name,
            accelerator=self.single_training_config.accelerator,
            precision=self.single_training_config.precision,
            do_pre_epoch_hook=self.single_training_config.do_pre_epoch_hook,
        )
        return best_model_path

    @staticmethod
    def _get_callbacks(
        early_stopping_cfg: dict, checkpoint_name: str, trial: optuna.Trial | None
    ) -> dict:
        es = EarlyStopping(**early_stopping_cfg)
        ckpt = ModelCheckpoint(
            dirpath=TRAINED_MODELS_DIR,
            monitor=early_stopping_cfg["monitor"],
            mode=early_stopping_cfg["mode"],
            save_top_k=1,
            filename=(
                f"{checkpoint_name}"
                + "-{epoch:02d}"
                + "-{"
                + early_stopping_cfg["monitor"]
                + ":.4f}"
            ),
        )
        callbacks = {"early_stopping": es}
        callbacks = (
            callbacks | {"model_checkpoint": ckpt} if trial is None else callbacks
        )
        pruning = (
            PyTorchLightningPruningCallback(trial, early_stopping_cfg["monitor"])
            if trial is not None
            else None
        )
        callbacks = (
            callbacks | {"optuna_pruning": pruning}
            if pruning is not None
            else callbacks
        )

        return callbacks

    def _get_log_to_wandb_as_config(
        self,
        log_type: Literal["optuna", "single"],
        architecture: dict,
        batch_size: int | None = None,
        learning_rate: float | None = None,
    ):
        config_mapper = {
            "optuna": self.optuna_config,
            "single": self.single_training_config,
        }
        config_obj = config_mapper[log_type]

        log_to_wandb_as_config = {
            "hparams": config_obj.hparams
            | self._get_additional_hparams_to_log(
                log_type=log_type, batch_size=batch_size, learning_rate=learning_rate
            ),
            "architecture": architecture,
            "early_stopping": config_obj.early_stopping,
        }
        return log_to_wandb_as_config

    def _get_additional_hparams_to_log(
        self,
        log_type: Literal["optuna", "single"],
        batch_size: int | None = None,
        learning_rate: float | None = None,
    ):
        configs_map = {
            "optuna": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "pruner": self.optuna_config.pruner,
                "sampler": self.optuna_config.sampler,
                "study_name": self.optuna_config.study_name,
            }
            if self.optuna_config is not None
            else {},
            "single": {
                "batch_size": self.single_training_config.dataloaders["batch_size"],
                "run_name": self.single_training_config.run_name,
            }
            if self.single_training_config is not None
            else {},
        }
        return configs_map[log_type]
