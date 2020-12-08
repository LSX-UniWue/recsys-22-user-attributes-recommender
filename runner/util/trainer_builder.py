from typing import Any, Dict, Union, Iterable

from pytorch_lightning import Trainer, Callback

TRAINER_INIT_KEYS = ['logger',
                     'checkpoint_callback',
                     'callbacks',
                     'default_root_dir',
                     'gradient_clip_val',
                     'process_position',
                     'num_nodes',
                     'num_processes',
                     'gpus',
                     'auto_select_gpus',
                     'tpu_cores',
                     'log_gpu_memory',
                     'progress_bar_refresh_rate',
                     'overfit_batches',
                     'track_grad_norm',
                     'check_val_every_n_epoch',
                     'fast_dev_run',
                     'accumulate_grad_batches',
                     'max_epochs',
                     'min_epochs',
                     'max_steps',
                     'min_steps',
                     'limit_train_batches',
                     'limit_val_batches',
                     'limit_test_batches',
                     'val_check_interval',
                     'flush_logs_every_n_steps',
                     'log_every_n_steps',
                     'accelerator',
                     'sync_batchnorm',
                     'precision',
                     'weights_summary',
                     'weights_save_path',
                     'num_sanity_val_steps',
                     'truncated_bptt_steps',
                     'resume_from_checkpoint',
                     'profiler',
                     'benchmark',
                     'deterministic',
                     'reload_dataloaders_every_epoch',
                     'auto_lr_find',
                     'replace_sampler_ddp',
                     'terminate_on_nan',
                     'auto_scale_batch_size',
                     'prepare_data_per_node',
                     'plugins',
                     'amp_backend',
                     'amp_level',
                     'distributed_backend',
                     'automatic_optimization']


class TrainerBuilder:

    def __init__(self, strict=False):
        self.kwargs = {}
        self.callbacks = []
        self.strict = strict

    def from_config(self, config: Dict[str, Any]):
        for key, value in config.items():
            self.set(key, value)
        return self

    def from_checkpoint(self, checkpoint_path: str):
        return self.set("resume_from_checkpoint", checkpoint_path)

    def set(self, key: str, value: Any):
        if key not in TRAINER_INIT_KEYS:
            if self.strict:
                raise KeyError(f"Invalid key '{key}' for Trainer constructor.")
        else:
            self.kwargs[key] = value

        return self

    def set_kw(self, **kwargs,):
        for key, value in kwargs.items():
            self.set(key, value)

        return self

    def add_callbacks(self, callbacks: Union[Callback, Iterable[Callback]]):
        self.callbacks += [callbacks] if type(callbacks) is Callback else callbacks
        return self

    def build(self) -> Trainer:
        return Trainer(**self.kwargs, callbacks=self.callbacks)
