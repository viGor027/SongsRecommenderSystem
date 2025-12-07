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
    AugmentedDataset,
    RamDataset,
)
from workflow_actions.paths import TRAINED_MODELS_DIR
from workflow_actions.json_handlers import write_dict_to_json
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from functools import partial
from torch.utils.data import DataLoader
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal


@dataclass
class RunSingleTrainingConfig:
    architecture: dict = field(default_factory=lambda: {})
    dataloaders: dict = field(
        default_factory=lambda: {
            "dataset_type": "ram_dataset",
            "batch_size": 32,
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "pin_memory": True,
            "drop_last": False,
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
    callbacks: dict = field(
        default_factory=lambda: {
            "add_model_checkpoint": True,
            "add_early_stopping": True,  # always True
        }
    )
    optimization: dict = field(
        default_factory=lambda: {
            "optimizer": "Adam",
            "optimizer_params": {"lr": 1e-3},
            "lr_schedule": None,
            "lr_schedule_params": None,
        }
    )
    trainer: dict = field(
        default_factory=lambda: {
            "max_epochs": 100,
            "accelerator": "auto",
            "precision": "32-true",
            "log_every_n_steps": 50,
            "enable_checkpointing": False,
        }
    )

    project: str = "default_project"
    run_name: str = "default_run"
    do_pre_epoch_hook: bool = False


@dataclass
class RunOptunaForAssembliesConfig:
    dataloaders: dict = field(
        default_factory=lambda: {
            "dataset_type": "ram_dataset",
            "batch_size": 32,
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "pin_memory": True,
            "drop_last": False,
        }
    )
    early_stopping: dict = field(
        default_factory=lambda: {
            "monitor": "val_loss",
            "mode": "min",
            "patience": 5,
        }
    )
    callbacks: dict = field(
        default_factory=lambda: {
            "add_model_checkpoint": False,
            "add_early_stopping": True,  # always True
            "add_pruning": True,
        }
    )
    optimization: dict = field(
        default_factory=lambda: {
            "optimizer": "Adam",
            "optimizer_params": {"lr": 1e-3},
            "lr_schedule": None,
            "lr_schedule_params": None,
        }
    )
    trainer: dict = field(
        default_factory=lambda: {
            "max_epochs": 100,
            "accelerator": "auto",
            "precision": "32-true",
            "log_every_n_steps": 50,
            "enable_checkpointing": False,
        }
    )
    optuna: dict = field(
        default_factory=lambda: {
            "n_trials": 5_000,
            "pruner": "median",
            "sampler": "tpe",
            "study_name": "default_study",
        }
    )
    project: str = "default_project"


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

        n_startup_trials = (
            int(self.optuna_config.optuna["n_trials"] * 0.15)
            if self.optuna_config is not None
            else 15
        )
        self.PRUNERS_MAP = {
            "median": partial(
                MedianPruner,
                n_startup_trials=n_startup_trials,
                n_warmup_steps=5,
                interval_steps=1,
            ),
        }

        self.SAMPLERS_MAP = {
            "tpe": partial(
                TPESampler,
                n_startup_trials=n_startup_trials,
                multivariate=True,
                group=True,
            ),
        }

        self._dataset_cache = {
            "train": None,
            "valid": None,
        }

    def get_dataloaders(
        self,
        dataset_type: Literal["ram_dataset", "augmented_dataset"],
        batch_size: int = 32,
        num_workers: int = 4,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        drop_last: bool = False,
    ):
        dataset_cls = {
            "ram_dataset": RamDataset,
            "augmented_dataset": AugmentedDataset,
        }[dataset_type]

        train_dataset = (
            dataset_cls(dataset_type="train")
            if self._dataset_cache["train"] is None
            else self._dataset_cache["train"]
        )
        valid_dataset = (
            dataset_cls(dataset_type="valid")
            if self._dataset_cache["valid"] is None
            else self._dataset_cache["valid"]
        )
        self._dataset_cache["train"] = train_dataset
        self._dataset_cache["valid"] = valid_dataset

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=dataset_cls.collate_concat,
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
            collate_fn=dataset_cls.collate_concat,
        )
        return train_loader, valid_loader

    @staticmethod
    def one_model_with_wandb(
        model,
        log_to_wandb_as_config: dict,
        optimization: dict,
        trainer: dict,
        train_dataloader,
        val_dataloader,
        callbacks: dict,
        project: str,
        run_name: str,
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
            optimization=optimization,
            do_pre_epoch_hook=do_pre_epoch_hook,
        )
        trainer = L.Trainer(
            logger=wandb_logger,
            callbacks=list(callbacks.values()),
            deterministic=True,
            **trainer,
        )

        try:
            trainer.fit(module, train_dataloader, val_dataloader)

            mc = callbacks.get("model_checkpoint")

            best_val = (
                mc.best_model_score.item()
                if mc is not None
                else callbacks["early_stopping"].best_score.item()
            )
            best_model_path = mc.best_model_path if mc is not None else None

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

            train_dataloader, val_dataloader = self.get_dataloaders(
                **self.optuna_config.dataloaders
            )

            log_to_wandb_as_config = {
                "architecture": assembly_cfg,
                "batch_size": self.optuna_config.dataloaders["batch_size"],
                "lr": self.optuna_config.optimization["optimizer_params"]["lr"],
            }

            run_name = f"{self.optuna_config.optuna['study_name']}_{assembly_type}_trial_{trial.number}"
            callbacks = self._get_callbacks(
                callbacks_config_from="optuna_cfg",
                **self.optuna_config.callbacks,
                checkpoint_name_prefix=self.optuna_config.optuna["study_name"],
                trial=trial,
            )

            ckpt_path, best_val = Train.one_model_with_wandb(
                model=model,
                log_to_wandb_as_config=log_to_wandb_as_config,
                optimization=self.optuna_config.optimization,
                trainer=self.optuna_config.trainer,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                callbacks=callbacks,
                project=self.optuna_config.project,
                run_name=run_name,
                do_pre_epoch_hook=False,
            )

            if best_model_path is None or best_val < best_score:
                best_score = best_val
                best_model_path = ckpt_path
            return best_val

        study = optuna.create_study(
            study_name=self.optuna_config.optuna["study_name"],
            direction="minimize",
            sampler=self.SAMPLERS_MAP[self.optuna_config.optuna["sampler"]](),
            pruner=self.PRUNERS_MAP[self.optuna_config.optuna["pruner"]](),
        )
        study.optimize(objective, n_trials=self.optuna_config.optuna["n_trials"])

        best_cfg = study.best_trial.params
        return study, best_cfg, best_model_path

    def run_optuna_hparams_search_for_single_architecture(self):
        """Uses RunSingleTrainingConfig"""
        if self.single_training_config is None:
            raise ValueError(
                "Running search for single architecture requires initializing Train "
                "object with same config as for run_single_training_config. "
                "Optimizer params, lr schedule and batch size will be overwritten by optuna."
            )
        best_model_path = None
        best_score = float("inf")

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_model_path, best_score

            model = self.model_initializer.get_model_assembly(
                assembly_config=self.single_training_config.architecture
            )

            suggested_training_hparams = (
                OptunaAssemblyConfigBuilder.suggest_training_hparams(trial=trial)
            )

            batch_size = suggested_training_hparams["batch_size"]
            self.single_training_config.dataloaders["batch_size"] = batch_size

            train_dataloader, val_dataloader = self.get_dataloaders(
                **self.single_training_config.dataloaders
            )

            log_to_wandb_as_config = {
                "architecture": self.single_training_config.architecture,
                "suggested_training_hparams": suggested_training_hparams,
                "dataloaders": self.single_training_config.dataloaders,
                "early_stopping": self.single_training_config.early_stopping,
                "trainer": self.single_training_config.trainer,
            }

            run_name = f"OptunaHparamSearch_{self.single_training_config.run_name}_trial_{trial.number}"
            callbacks = self._get_callbacks(
                callbacks_config_from="single_cfg",
                **self.single_training_config.callbacks,
                checkpoint_name_prefix=self.single_training_config.run_name,
                add_pruning=True,
                trial=trial,
            )
            ckpt_path, best_val = Train.one_model_with_wandb(
                model=model,
                log_to_wandb_as_config=log_to_wandb_as_config,
                optimization=suggested_training_hparams["optimization"],
                trainer=self.single_training_config.trainer,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                callbacks=callbacks,
                project=self.single_training_config.project,
                run_name=run_name,
                do_pre_epoch_hook=self.single_training_config.do_pre_epoch_hook,
            )

            if best_model_path is None or best_val < best_score:
                best_score = best_val
                best_model_path = ckpt_path
            return best_val

        study = optuna.create_study(
            study_name=f"OptunaHparamSearch_{self.single_training_config.run_name}",
            direction="minimize",
            sampler=self.SAMPLERS_MAP["tpe"](),
            pruner=self.PRUNERS_MAP["median"](),
        )
        study.optimize(objective, n_trials=100)

        best_cfg = study.best_trial.params
        write_dict_to_json(
            data={
                "best_model_path": best_model_path,
                "best_cfg": best_cfg,
            },
            file_path=TRAINED_MODELS_DIR
            / f"OptunaHparamSearch_{self.single_training_config.run_name}_summary.json",
        )
        return study, best_cfg, best_model_path

    def run_single_training(self) -> tuple[None | str, float]:
        if self.single_training_config is None:
            raise ValueError(
                "Running single training requires initializing Train object with run_single_training_config."
            )
        model = self.model_initializer.get_model_assembly(
            assembly_config=self.single_training_config.architecture
        )
        log_to_wandb_as_config = asdict(self.single_training_config)

        train_dataloader, val_dataloader = self.get_dataloaders(
            **self.single_training_config.dataloaders
        )
        callbacks = self._get_callbacks(
            callbacks_config_from="single_cfg",
            **self.single_training_config.callbacks,
            checkpoint_name_prefix=self.single_training_config.run_name,
            add_pruning=False,
            trial=None,
        )
        ckpt_path, best_val = Train.one_model_with_wandb(
            model=model,
            log_to_wandb_as_config=log_to_wandb_as_config,
            optimization=self.single_training_config.optimization,
            trainer=self.single_training_config.trainer,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            callbacks=callbacks,
            project=self.single_training_config.project,
            run_name=self.single_training_config.run_name,
            do_pre_epoch_hook=self.single_training_config.do_pre_epoch_hook,
        )
        return ckpt_path, best_val

    def _get_callbacks(
        self,
        callbacks_config_from: Literal["optuna_cfg", "single_cfg"],
        checkpoint_name_prefix: str,
        add_model_checkpoint: bool,
        add_early_stopping: bool,
        add_pruning: bool = False,
        trial: optuna.Trial | None = None,
    ) -> dict:
        config_obj = {
            "optuna_cfg": self.optuna_config,
            "single_cfg": self.single_training_config,
        }[callbacks_config_from]
        monitor_value = config_obj.early_stopping["monitor"]

        es = EarlyStopping(**config_obj.early_stopping)
        ckpt = ModelCheckpoint(
            dirpath=TRAINED_MODELS_DIR,
            monitor=monitor_value,
            mode=config_obj.early_stopping["mode"],
            save_top_k=1,
            save_weights_only=True,
            filename=(
                f"{checkpoint_name_prefix}"
                + "-{epoch:02d}"
                + "-{"
                + monitor_value
                + ":.4f}"
            ),
        )
        pruning = (
            PyTorchLightningPruningCallback(trial, monitor_value)
            if trial is not None
            else None
        )

        callbacks = {"early_stopping": es} if add_early_stopping else {}
        callbacks = (
            callbacks | {"model_checkpoint": ckpt}
            if add_model_checkpoint
            else callbacks
        )
        callbacks = (
            callbacks | {"optuna_pruning": pruning}
            if (trial is not None and add_pruning)
            else callbacks
        )
        return callbacks
