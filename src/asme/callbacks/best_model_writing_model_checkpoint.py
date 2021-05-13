import os, errno
from pathlib import Path
from typing import Union, Optional, Dict, Any

import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import LightningModule


class BestModelWritingModelCheckpoint(ModelCheckpoint):
    """
        Decorates the ModelCheckpoint class by calling the to_yaml method after each epoch and creating a symlink
        pointing to the best model. If output_base_path (path for the best-k-models file) can not be found in the
        configuration, the dirpath of model_checkpoint is used as the default. Same goes for output_filename where
        the default value is "best_k_models.yaml".
    """

    def __init__(self, model_checkpoint: ModelCheckpoint, output_base_path: str, output_filename: str):
        super().__init__()
        self.target_object = model_checkpoint
        if output_base_path is not None:
            self.output_base_path = output_base_path
        else:
            self.output_base_path = self.target_object.dirpath
        if output_filename is not None:
            self.output_filename = output_filename
        else:
            self.output_filename = "best_k_models.yaml"

    def on_train_end(self, trainer, pl_module: LightningModule):
        self.target_object.on_train_end(trainer, pl_module)
        self.target_object.to_yaml(os.path.join(self.output_base_path, self.output_filename))
        try:
            os.symlink(self.target_object.best_model_path, os.path.join(self.output_base_path, "best.ckpt"))
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(os.path.join(self.output_base_path, "best.ckpt"))
                os.symlink(self.best_model_path, os.path.join(self.output_base_path, "best.ckpt"))

    def on_pretrain_routine_start(self, trainer, pl_module):
        self.target_object.on_pretrain_routine_start(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        self.target_object.on_validation_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        return self.target_object.on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]):
        self.target_object.on_load_checkpoint(checkpointed_state)

    def save_checkpoint(self, trainer, pl_module):
        self.target_object.save_checkpoint(trainer, pl_module)

    def _del_model(self, filepath: str):
        return self.target_object._del_model(filepath)

    def _save_model(self, filepath: str, trainer, pl_module):
        return self.target_object._save_model(filepath, trainer, pl_module)

    def check_monitor_top_k(self, trainer, current: Optional[torch.Tensor] = None) -> bool:
        return self.target_object.check_monitor_top_k(current)

    @classmethod
    def _format_checkpoint_name(cls, filename: Optional[str], epoch: int, step: int, metrics: Dict[str, Any],
                                prefix: str = "") -> str:
        return super()._format_checkpoint_name(filename, epoch, step, metrics, prefix)

    def format_checkpoint_name(self, epoch: int, step: int, metrics: Dict[str, Any], ver: Optional[int] = None) -> str:
        return self.target_object.format_checkpoint_name(epoch, step, metrics, ver)

    def _add_backward_monitor_support(self, trainer):
        self.target_object._add_backward_monitor_support(trainer)

    def _validate_monitor_key(self, trainer):
        return self.target_object._validate_monitor_key(trainer)

    def _get_metric_interpolated_filepath_name(self, ckpt_name_metrics: Dict[str, Any], epoch: int, step: int, trainer,
                                               del_filepath: Optional[str] = None) -> str:
        return self.target_object._get_metric_interpolated_filepath_name(ckpt_name_metrics, epoch, step, trainer,
                                                                         del_filepath)

    def _monitor_candidates(self, trainer):
        return self.target_object._monitor_candidates(trainer)

    def _save_last_checkpoint(self, trainer, pl_module, ckpt_name_metrics):
        self.target_object._save_last_checkpoint(trainer, pl_module, ckpt_name_metrics)

    def _save_top_k_checkpoints(self, trainer, pl_module, metrics):
        self.target_object._save_top_k_checkpoints(trainer, pl_module, metrics)

    def _is_valid_monitor_key(self, metrics):
        return self.target_object._is_valid_monitor_key(metrics)

    def _update_best_and_save(self, current: torch.Tensor, epoch: int, step: int, trainer, pl_module,
                              ckpt_name_metrics):
        self.target_object._update_best_and_save(current, epoch, step, trainer, pl_module, ckpt_name_metrics)

    def to_yaml(self, filepath: Optional[Union[str, Path]] = None):
        self.target_object.to_yaml(filepath)

    def file_exists(self, filepath: Union[str, Path], trainer) -> bool:
        return self.target_object.file_exists(filepath, trainer)
