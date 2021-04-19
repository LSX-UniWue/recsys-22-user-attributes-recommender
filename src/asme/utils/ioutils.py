import json
import os
from pathlib import Path
from typing import List, Callable, Any

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LoggerCollection, LightningLoggerBase

from asme.init.config import Config


PROCESSED_CONFIG_NAME = "config.jsonnet"
FINISHED_FLAG_NAME = ".FINISHED"


def save_config(config: Config,
                path: Path,
                make_parents=True
                ):
    """
    Saves the provided configuration at the specified path, optionally creating parent directories.
    """
    if make_parents:
        os.makedirs(path,exist_ok=True)
    full_path = path / PROCESSED_CONFIG_NAME
    with open(full_path, "w") as f:
        json.dump(config.config, f, indent=4)


def _build_log_dir_from_logger(logger: LightningLoggerBase) -> Path:
    # FIXME: remove method as soon as the is a common build method for this in the lightning lib
    log_dir = Path(logger.save_dir)
    logger_name = logger.name
    if logger_name and len(logger_name) > 0:
        log_dir = log_dir / logger_name

    logger_version = logger.version

    if logger_version is not None and len(logger_version) > 0:
        logger_version_subpath = logger_version if isinstance(logger_version, str) else f"version_{logger_version}"
        log_dir = log_dir / logger_version_subpath

    return log_dir


def determine_log_dir(trainer: Trainer
                      ) -> Path:
    """
    Determines the logger directory used by the trainer in the following way:
    1. If multiple loggers are specified the save directory of the first one is used.
    2. If only a singe logger is registered, its log directory is used.
    3. If the provided trainer has no attached loggers, its default_root_directory is used.
    """
    if trainer.logger is None:
        return Path(trainer.default_root_dir)
    if type(trainer.logger) is LoggerCollection:
        # FIXME: the first logger can have no save_dir
        return _build_log_dir_from_logger(trainer.logger[0])

    return _build_log_dir_from_logger(trainer.logger)


def save_finished_flag(path: Path,
                       make_parents=True
                       ):
    """
    Saves an empty file named '.FINISHED' at the specified path, optionally creating parent directories.
    """
    if make_parents:
        os.makedirs(path, exist_ok=True)
    full_path = path / FINISHED_FLAG_NAME
    with open(full_path, "w"):
        pass


def finished_flag_exists(path: Path) -> bool:
    """
    Checks whether a finished flag, i.e a file named '.FINISHED' exists at the provided path.
    """
    full_path = path / FINISHED_FLAG_NAME
    return os.path.isfile(full_path)


def find_all_files(base_path: Path,
                   suffix: str,
                   follow_links: bool = True
                   ) -> List[Path]:
    all_files = []
    for root, dirs, files in os.walk(str(base_path), followlinks=follow_links):
        for file in files:
            if file.endswith(suffix):
                all_files.append(Path(os.path.join(root, file)))

    return all_files


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
