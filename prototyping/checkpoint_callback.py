from cloud.cloud_utils import save_checkpoint_to_gcs
from lightning.pytorch.callbacks import Callback


class GCSModelCheckpoint(Callback):
    def __init__(self, bucket_name, folder_name, checkpoint_name, monitor="val_loss", mode="min"):
        """
        Custom Callback for saving the best model checkpoint to GCS.

        Args:
            bucket_name (str): GCS bucket name.
            folder_name (str): Folder in GCS where the checkpoint should be saved.
            checkpoint_name (str): Name of the checkpoint file (without extension).
            monitor (str): Metric to monitor for determining the "best" checkpoint.
            mode (str): One of {"min", "max"}. Whether lower or higher metric values are better.
        """
        super().__init__()
        self.bucket_name = bucket_name
        self.folder_name = folder_name
        self.checkpoint_name = checkpoint_name
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else -float("inf")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """
        Called when a checkpoint is saved.

        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being trained.
            checkpoint: The checkpoint dictionary.
        """
        try:
            current_value = trainer.callback_metrics.get(self.monitor)
            if current_value is None:
                print(f"Warning: Metric {self.monitor} not found in callback metrics.")
                return

            is_better = (
                current_value < self.best_value if self.mode == "min" else current_value > self.best_value
            )
            if is_better:
                self.best_value = current_value
                checkpoint_dict = checkpoint
                checkpoint_dict['best_loss'] = self.best_value
                success = save_checkpoint_to_gcs(
                    checkpoint=checkpoint_dict,
                    bucket_name=self.bucket_name,
                    folder_name=self.folder_name,
                    checkpoint_name=self.checkpoint_name
                )
                if success:
                    print(f"Best model checkpoint saved to {self.folder_name}/{self.checkpoint_name}.ckpt in {self.bucket_name}")
            else:
                print(f"Skipping checkpoint save, {self.monitor} did not improve.")
        except Exception as e:
            print(f"Error saving checkpoint to GCS: {e}")

    def state_dict(self):
        """
        Return the state of the callback (e.g., best_value).
        """
        return {'best_value': self.best_value}

    def load_state_dict(self, state_dict):
        """
        Restore the state of the callback (e.g., best_value).
        """
        self.best_value = state_dict.get('best_value', float("inf") if self.mode == "min" else -float("inf"))

    def retrieve_best_loss(self):
        return self.best_value
