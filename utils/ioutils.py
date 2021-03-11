import json
import os
from pathlib import Path
from typing import List, Callable, Any

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection

from init.config import Config


PROCESSED_CONFIG_NAME = "config.jsonnet"
FINISHED_FLAG_NAME = ".FINISHED"


def save_config(config: Config, path: Path, make_parents=True):
    """
    Saves the provided configuration at the specified path, optionally creating parent directories.
    """
    if make_parents:
        os.makedirs(path,exist_ok=True)
    full_path = path / PROCESSED_CONFIG_NAME
    with open(full_path, "w") as f:
        json.dump(config.config, f, indent=4)


def determine_log_dir(trainer: Trainer) -> str:
    """
    Determines the logger directory used by the trainer in the following way:
    1. If multiple loggers are specified the save directory of the first one is used.
    2. If only a singe logger is registered, its save directory is used.
    3. If the provided trainer has no attached loggers, its default_root_directory is used.
    """
    if trainer.logger is None:
        return trainer.default_root_dir
    if type(trainer.logger) is LoggerCollection:
        return trainer.logger[0].save_dir

    return trainer.logger.save_dir


def save_finished_flag(path: Path, make_parents=True):
    """
    Saves an empty file named '.FINISHED' at the specified path, optionally creating parent directories.
    """
    if make_parents:
        os.makedirs(path,exist_ok=True)
    full_path = path / FINISHED_FLAG_NAME
    with open(full_path, "w"):
        pass


def finished_flag_exists(path: Path) -> bool:
    """
    Checks whether a finished flag, i.e a file named '.FINISHED' exists at the provided path.
    """
    full_path = path / FINISHED_FLAG_NAME
    return os.path.isfile(full_path)


def load_file_with_item_ids(path: Path) -> List[int]:
    """
    loads a file containing item ids into a list
    :param path: the path of the file
    :return:
    """
    items = _load_file_line_my_line(path, int)
    sorted(items)
    return items


def _load_file_line_my_line(path: Path, line_converter: Callable[[str], Any]) -> List[Any]:
    with open(path) as item_file:
        return [line_converter(line) for line in item_file.readlines()]
