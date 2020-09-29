from pyhocon import ConfigTree
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class ConfigTrainerBuilder:

    def build(self, config: ConfigTree):
        trainer_config = config["trainer"]

        max_epochs = trainer_config.get_int("max_epochs", 10)
        val_check_interval = trainer_config.get_float("val_check_interval", 1.0)
        limit_train_batches = trainer_config.get_float("limit_train_batches", 1.0)
        limit_val_batches = trainer_config.get_float("limit_val_batches", 1.0)
        limit_test_batches = trainer_config.get_float("limit_test_batches", 1.0)
        default_root_dir = trainer_config.get_string("default_root_dir", None)
        gpus = trainer_config.get_int("gpus", None)

        parameters = {
            "max_epochs": max_epochs,
            "val_check_interval": val_check_interval,
            "limit_train_batches": limit_train_batches,
            "limit_val_batches": limit_val_batches,
            "limit_test_batches": limit_test_batches,
            "default_root_dir": default_root_dir,
            "gpus": gpus
        }

        if "checkpoint_callback" in trainer_config:
            parameters["checkpoint_callback"] = self.get_checkpoint_callback(trainer_config["checkpoint_callback"])

        trainer = pl.Trainer(**parameters)

        return trainer

    def get_checkpoint_callback(self, config: ConfigTree) -> ModelCheckpoint:
        parameters = {
            "monitor": config.get_string("monitor", "loss"),
            "filepath": config.get_string("filepath", None),
            "save_top_k": config.get_int("save_top_k", 3),
            "save_last": config.get_bool("save_last", True)
        }

        mc = ModelCheckpoint(**parameters)

        return mc
