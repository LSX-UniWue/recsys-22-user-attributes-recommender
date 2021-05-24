import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List

import pytorch_lightning.core as pl
from tqdm import tqdm

from data.datamodule.config import AsmeDataModuleConfig
from data.datamodule.preprocessing import PreprocessingAction, EXTRACTED_DIRECTORY_KEY
from data.datamodule.unpacker import Unpacker
from data.datasets.config import get_movielens_1m_config
from datasets.dataset_pre_processing.utils import download_dataset


class AsmeDataModule(pl.LightningDataModule):

    PREPROCESSING_FINISHED_FLAG = ".PREPROCESSING_FINISHED"

    def __init__(self, config: AsmeDataModuleConfig):
        super().__init__()
        self.config = config

    def prepare_data(self):
        dsConfig = self.config.datasetConfig

        # Check whether we already preprocessed the dataset
        if self._check_finished_flag(dsConfig.location):
            return

        if dsConfig.url is not None:
            print(f"Downloading dataset...")
            dataset_file = download_dataset(dsConfig.url, dsConfig.location)
        else:
            print(f"No download URL specified, using local copy at '{dsConfig.location}'")
            dataset_file = dsConfig.location

        # If necessary, unpack the dataset
        if dsConfig.unpacker is not None:
            print(f"Unpacking dataset...", end="")
            dsConfig.unpacker(dataset_file)
            print("Done.")

        # Apply preprocessing steps
        for i, step in enumerate(dsConfig.preprocessing_actions):
            print(f"Applying preprocessing step '{step.name()}' ({i+1}/{len(dsConfig.preprocessing_actions)})...", end="")
            step.apply(dsConfig.context)
            print("Done.")

        # Write finished flag
        (dsConfig.location / self.PREPROCESSING_FINISHED_FLAG).touch()

    def setup(self, stage: Optional[str] = None):
        # This should essentially do the job of the data sources factory
        pass

    def _check_finished_flag(self, directory: Path) -> bool:
        return os.path.exists(directory / self.PREPROCESSING_FINISHED_FLAG)


if __name__ == "__main__":
    dSconfig = get_movielens_1m_config(Path("/tmp/ml-1m"), Path("/tmp/ml-1m/raw"),min_sequence_length=100, min_item_feedback=50)
    config = AsmeDataModuleConfig(dSconfig)
    module = AsmeDataModule(config)
    module.prepare_data()