import os, errno
from datetime import timedelta
from pathlib import Path
from typing import Optional, Dict, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


class BestModelWritingModelCheckpoint(ModelCheckpoint):
    """
        Tracks best checkpoints by decorating the `ModelCheckpoint` class.

        Calls the to_yaml method after each epoch and creates a symlink `best.ckpt` pointing to the best model.
        Additionally, validation metrics for each checkpoint are stored in `<output_base_path>/<output_filename>`.
        If output_base_path can not be found in the configuration, the dirpath of model_checkpoint is used as the default.
        Same goes for output_filename where the default value is "best_k_models.yaml".
    """
    # FIXME: I think we can remove the first three parameters, since they can be fixed or inferred
    def __init__(self,
                 output_base_path: str,
                 output_filename: str,
                 symlink_name: str,
                 dirpath: Optional[Union[str, Path]] = None,
                 filename: Optional[str] = None,
                 monitor: Optional[str] = None,
                 verbose: bool = False,
                 save_last: Optional[bool] = None,
                 save_top_k: int = 1,
                 save_weights_only: bool = False,
                 mode: str = "min",
                 auto_insert_metric_name: bool = True,
                 every_n_train_steps: Optional[int] = None,
                 train_time_interval: Optional[timedelta] = None,
                 every_n_epochs: Optional[int] = None,
                 save_on_train_epoch_end: Optional[bool] = None,
                 period: Optional[int] = None,
                 every_n_val_epochs: Optional[int] = None,
                 ):
        super().__init__(dirpath,
                         filename,
                         monitor,
                         verbose,
                         save_last,
                         save_top_k,
                         save_weights_only,
                         mode,
                         auto_insert_metric_name,
                         every_n_train_steps,
                         train_time_interval,
                         every_n_epochs,
                         save_on_train_epoch_end,
                         period,
                         every_n_val_epochs)

        if output_base_path is not None:
            self.output_base_path = Path(output_base_path)
        else:
            self.output_base_path = Path(self.dirpath)

        if not self.output_base_path.exists():
            self.output_base_path.mkdir(parents=True)

        if output_filename is not None:
            self.output_filename = output_filename
        else:
            self.output_filename = "best_k_models.yaml"

        self.symlink_name = symlink_name

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        super().to_yaml(os.path.join(self.output_base_path, self.output_filename))

        # Save a symlink to the best model checkpoint
        self._save_best_model_checkpoint_symlink()

        # Also save the raw weights of the current best model
        self._save_best_checkpoint_weights_only(trainer)

    def _save_best_model_checkpoint_symlink(self):
        symlink_path = self.output_base_path.joinpath(self.symlink_name)
        # here we only link relative paths, to prevent wrong links when
        # the result path is mounted into a VM, container …
        best_checkpoint_path = Path(self.best_model_path).name
        try:
            symlink_path.symlink_to(best_checkpoint_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                symlink_path.unlink()
                symlink_path.symlink_to(best_checkpoint_path)

    def _save_best_checkpoint_weights_only(self, trainer: "pl.Trainer"):
        # This just figures out whether a file extension was already provided with the symlink name and uses it if so
        ext = Path(self.symlink_name).suffix
        if ext is "":
            best_checkpoint_weights_only_name = f"{self.symlink_name}-weights-only.ckpt"
        else:
            name = Path(self.symlink_name).stem
            # The dot between the file name and extension is already included in "ext"
            best_checkpoint_weights_only_name = f"{name}-weights-only{ext}"

        best_checkpoint_weights_only_path = self.output_base_path.joinpath(best_checkpoint_weights_only_name)

        # If we have saved a weights-only best checkpoint previously, remove it
        if best_checkpoint_weights_only_path.exists():
            best_checkpoint_weights_only_path.unlink()

        # Save the checkpoint using the trainer directly
        trainer.save_checkpoint(best_checkpoint_weights_only_path, weights_only=True)

    @classmethod
    def _format_checkpoint_name(cls,
                                filename: Optional[str],
                                metrics: Dict,
                                prefix: str = "",
                                auto_insert_metric_name: bool = True,
                                ) -> str:
        return super()._format_checkpoint_name(filename, metrics, prefix, auto_insert_metric_name)

