from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Union, Dict

from torch.utils.data import DataLoader

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.init.factories.data_sources.loader import LoaderFactory
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.init.templating.datasources.datasources import Stage, DatasetSplit, DatasetBuilder, \
    SequenceDatasetRatioSplitBuilder, LeaveOneOutSessionDatasetBuilder, _transfer_properties, \
    NextPositionDatasetBuilder, TARGET_EXTRACTOR_PROCESSOR_CONFIG


class TemplateDataSourcesFactory(ObjectFactory):

    def __init__(self, key: str):
        super().__init__()
        self._key = key
        self._factory = ConditionalFactory(key, {
            "masked": MaskTemplateDataSourcesFactory()
        }, "data_sources", ["data_sources"])

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self._factory.can_build(config, context)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        return self._factory.build(config, context)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ["data_sources"]

    def config_key(self) -> str:
        return "data_sources"


class BaseTemplateDataSourcesFactory(ObjectFactory):

    def __init__(self):
        super().__init__()
        self._loader_factory = LoaderFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Dict[str, DataLoader]:
        return {
            "train": self._build_train_datasource(config, context),
            "validation": self._build_validation_datasource(config, context),
            "test": self._build_test_datasource(config, context)
        }

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return ""

    def _build_datasource(self, config: Config, context: Context) -> DataLoader:
        return self._loader_factory.build(config, context)

    @abstractmethod
    def _build_train_datasource(self, config: Config, context: Context) -> DataLoader:
        pass

    @abstractmethod
    def _build_validation_datasource(self, config: Config, context: Context) -> DataLoader:
        pass

    @abstractmethod
    def _build_test_datasource(self, config: Config, context: Context) -> DataLoader:
        pass


class MaskTemplateDataSourcesFactory(BaseTemplateDataSourcesFactory):

    TRAIN_DATASET_BUILDERS = [SequenceDatasetRatioSplitBuilder(), LeaveOneOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def __init__(self):
        super().__init__()

    def _build_train_datasource(self, config: Config, context: Context) -> DataLoader:
        mask_probability = config.get_or_default('mask_probability', 0.2)
        only_last_item_mask_prob = config.get_or_default('only_last_item_mask_prob', 0.1)
        cloze_processor = {
            'type': "cloze",
            'mask_probability': mask_probability,
            'only_last_item_mask_prob': only_last_item_mask_prob,
        }
        loader_config = build_default_loader_config(config, Stage.TRAIN, self.TRAIN_DATASET_BUILDERS, [cloze_processor])
        return self._build_datasource(loader_config, context)

    def _build_validation_datasource(self, config: Config, context: Context) -> DataLoader:
        mask_last_item_processor = {
            'type': 'last_item_mask'
        }
        loader_config = build_default_loader_config(config, Stage.VALIDATION, self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG, mask_last_item_processor])
        return self._build_datasource(loader_config, context)

    def _build_test_datasource(self, config: Config, context: Context) -> DataLoader:
        mask_last_item_processor = {
            'type': 'last_item_mask'
        }
        loader_config = build_default_loader_config(config, Stage.VALIDATION, self.TEST_VALID_DATASET_BUILDERS,
                                                    [TARGET_EXTRACTOR_PROCESSOR_CONFIG, mask_last_item_processor])
        return self._build_datasource(loader_config, context)


def build_dataset_config(dataset_builders: List[DatasetBuilder], config: Config, stage: Stage, additional_processors: List[Dict[str, Any]] = None) -> Dict[str, Any]:

    if additional_processors is None:
        additional_processors = []

    processors =[{
        "type": "tokenizer"
    }] + additional_processors

    split = DatasetSplit[config.get("split").upper()]

    def _build_dataset_config() -> Dict[str, Any]:
        for datasource_builder in dataset_builders:
            if datasource_builder.can_build_dataset_definition(split):
                return datasource_builder.build_dataset_definition(stage, config.config)
        raise ValueError('no datasource builder found')

    datasource_config = _build_dataset_config()
    datasource_config["processors"] = processors

    return datasource_config


def build_default_loader_config(config: Config, stage: Stage, dataset_builders: List[DatasetBuilder], processors: List[Dict[str, Any]] = None) -> Config:

    dataset_config = build_dataset_config(dataset_builders, config, stage, processors)

    base_batch_size = config.get_or_default('batch_size', 8)
    batch_size = config.get_or_default(f'{stage.value}_batch_size', base_batch_size)
    shuffle = config.get_or_default('shuffle', stage == Stage.TRAIN)

    loader_config = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "dataset": dataset_config
    }

    loader_config = _transfer_properties(config.config, loader_config,
                                              ['max_seq_step_length', 'num_workers', 'dynamic_padding'])

    return Config(loader_config)
