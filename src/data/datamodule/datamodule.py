import os
import pickle
import shutil

from pathlib import Path
from typing import Optional, List, Union

import pytorch_lightning.core as pl
from torch.utils.data import DataLoader

from asme.init.context import Context
from asme.init.factories.data_sources.template_datasources import TemplateDataSourcesFactory
from asme.init.factories.data_sources.user_defined_datasources import UserDefinedDataSourcesFactory
from asme.init.templating.datasources.datasources import DatasetSplit
from asme.utils.logging import get_logger
from data import BASE_DATASET_PATH_CONTEXT_KEY, CURRENT_SPLIT_PATH_CONTEXT_KEY, DATASET_PREFIX_CONTEXT_KEY
from data.datamodule.checkpoint import PreprocessingCheckpoint
from data.datamodule.config import AsmeDataModuleConfig
from data.datamodule.metadata import DatasetMetadata
from datasets.dataset_pre_processing.utils import download_dataset

logger = get_logger(__name__)


class AsmeDataModule(pl.LightningDataModule):
    PREPROCESSING_FINISHED_FLAG = ".METADATA"
    CHECKPOINT_NAME = ".CHECKPOINT"

    def __init__(self, config: AsmeDataModuleConfig, context: Context = Context()):
        super().__init__()
        self.config = config
        self.context = context

        self._objects = {}
        self._has_setup = False

    @property
    def has_setup(self):
        return self._has_setup

    def prepare_data(self):
        ds_config = self.config.dataset_preprocessing_config

        # Check whether we already preprocessed the dataset
        if self._check_preprocessing_finished():
            metadata = self._load_metadata()
            logger.info("Found a finished flag in the target directory. Assuming the dataset is already preprocessed.")
        else:
            logger.info("Preprocessing dataset:")

            first_step = 0
            if (checkpoint := self._load_checkpoint()) is not None and not self.config.force_regeneration:
                logger.info(f"Found a checkpoint for step {checkpoint.step + 1}. Continuing from there.")
                first_step = checkpoint.step + 1
                ds_config.context = checkpoint.context
            else:
                if ds_config.url is not None:
                    logger.info(
                        f"Downloading dataset {self.config.dataset} from {self.config.dataset_preprocessing_config.url}.")
                    dataset_file = download_dataset(ds_config.url, ds_config.location)
                else:
                    logger.info(f"No download URL specified, using local copy at '{ds_config.location}'.")
                    dataset_file = ds_config.location

                # If necessary, unpack the dataset
                if ds_config.unpacker is not None:
                    logger.info(f"Unpacking dataset.")
                    ds_config.unpacker(dataset_file)

            # Apply preprocessing steps
            actions_left = ds_config.preprocessing_actions[first_step:]
            for i, step in enumerate(actions_left):
                logger.info(
                    f"Applying preprocessing step '{step.name()}' ({i + first_step + 1}/{len(ds_config.preprocessing_actions)})")
                if step.dry_run_available(ds_config.context) and not self.config.force_regeneration:
                    logger.info(f"Skipping this step since dry run is available.")
                step.apply(ds_config.context, self.config.force_regeneration)
                checkpoint = PreprocessingCheckpoint(i, ds_config.context)
                checkpoint.save(self.config.dataset_preprocessing_config.location / self.CHECKPOINT_NAME)

            # Save dataset metadata
            metadata = DatasetMetadata.from_context(ds_config.context)
            self._write_metadata(metadata)

            # Remove the checkpoint after we are done processing
            os.remove(self.config.dataset_preprocessing_config.location / self.CHECKPOINT_NAME)

        # Populate context with the dataset path
        self.context.set(BASE_DATASET_PATH_CONTEXT_KEY, self.config.dataset_preprocessing_config.location)
        split = self._determine_split()
        split_path = metadata.ratio_path if split == DatasetSplit.RATIO_SPLIT else metadata.loo_path
        self.context.set(CURRENT_SPLIT_PATH_CONTEXT_KEY, split_path)
        # Also put the prefix into the context
        self.context.set(DATASET_PREFIX_CONTEXT_KEY, self.config.dataset)

    def setup(self, stage: Optional[str] = None):
        if len(msg := self._validate_config()) > 0:
            raise KeyError(f"Invalid config: {msg}.")

        if self.config.cache_path is not None:
            # Copy the dataset
            ds_location = self.config.dataset_preprocessing_config.location
            logger.info(f"Caching dataset from '{ds_location}' to '{self.config.cache_path}'.")
            shutil.copytree(ds_location, self.config.cache_path, dirs_exist_ok=True)

            # adjust the context values such that the factories infer correct paths
            self.context.set(BASE_DATASET_PATH_CONTEXT_KEY, self.config.cache_path, overwrite=True)
            self._adjust_split_path_for_caching(CURRENT_SPLIT_PATH_CONTEXT_KEY)

        # Build the datasources depending on what the user specified
        if self.config.template is not None:
            factory = TemplateDataSourcesFactory("name")
            self._objects = factory.build(self.config.template, self.context)
        else:
            factory = UserDefinedDataSourcesFactory()
            self._objects = factory.build(self.config.data_sources, self.context)

        self._has_setup = True

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader("validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader("test")

    def _get_dataloader(self, name: str):
        if not self.has_setup:
            self.setup()
        return self._objects[name]

    def _check_preprocessing_finished(self) -> bool:
        return os.path.exists(
            self.config.dataset_preprocessing_config.location / self.PREPROCESSING_FINISHED_FLAG)

    def _load_checkpoint(self) -> Optional[PreprocessingCheckpoint]:
        checkpoint_path = self.config.dataset_preprocessing_config.location / self.CHECKPOINT_NAME
        if os.path.exists(checkpoint_path):
            try:
                return PreprocessingCheckpoint.load(checkpoint_path)
            except:
                logger.warning("Failed to parse the checkpoint file. Removing it and starting over.")
                os.remove(checkpoint_path)
        else:
            return None

    def _write_metadata(self, metadata: DatasetMetadata):
        metadata_path = self.config.dataset_preprocessing_config.location / self.PREPROCESSING_FINISHED_FLAG
        with open(metadata_path, "w") as f:
            f.write(metadata.to_json())

    def _load_metadata(self) -> DatasetMetadata:
        metadata_path = self.config.dataset_preprocessing_config.location / self.PREPROCESSING_FINISHED_FLAG
        with open(metadata_path, "r") as f:
            return DatasetMetadata.from_json(f.read())

    def _determine_split(self) -> Optional[DatasetSplit]:
        if self.config.data_sources is not None:
            split = self.config.data_sources.get("split")
        else:
            split = self.config.template.get("split")

        if split is None:
            return None
        else:
            return DatasetSplit[split.upper()]

    def _adjust_split_path_for_caching(self, key: Union[str, List[str]]):
        """
        Adjusts the split path stored in the context with the specified key to point to the cache directory instead.
        """
        current_value = self.context.get(key)
        split_dir = os.path.split(current_value)[-1]
        self.context.set(key, os.path.join(self.config.cache_path, split_dir), overwrite=True)

    def _validate_config(self) -> List[str]:
        errors = []
        if self.config.template is not None and self.config.data_sources is not None:
            errors.append("Please specify one of 'template' or 'data_sources'")

        if self._determine_split() is None:
            errors.append("Please specify a split type via the 'split' attribute. (Either 'leave_one_out' or 'ratio').")

        return errors
