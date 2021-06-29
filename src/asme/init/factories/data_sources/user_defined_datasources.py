import copy
from typing import List, Union, Any, Dict

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.init.factories.data_sources.common import build_default_loader_config
from asme.init.factories.data_sources.loader import LoaderFactory
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.init.templating.datasources.datasources import Stage, DatasetSplit, NextPositionDatasetBuilder, \
    SequenceDatasetRatioSplitBuilder, NextPositionWindowDatasetBuilder, LeaveOneOutSessionDatasetBuilder, \
    LeaveOneOutSequenceWindowDatasetBuilder, LeaveOneOutNextPositionDatasetBuilder, TARGET_EXTRACTOR_PROCESSOR_CONFIG


class UserDefinedDataSourcesFactory(ObjectFactory):
    DATASET_BUILDER = {
        DatasetSplit.RATIO_SPLIT: {
            "next_item": NextPositionDatasetBuilder(),
            "session": SequenceDatasetRatioSplitBuilder(),
            "window": NextPositionWindowDatasetBuilder()
        },
        DatasetSplit.LEAVE_ONE_OUT: {
            "next_item": LeaveOneOutNextPositionDatasetBuilder(),
            "session": LeaveOneOutSessionDatasetBuilder(),
            "window": LeaveOneOutSequenceWindowDatasetBuilder()
        }
    }

    def __init__(self):
        super().__init__()
        self._factory = LoaderFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        for stage in Stage:
            result = self._factory.can_build(config.get_config(stage.value), context)
            if result.type != CanBuildResultType.CAN_BUILD:
                return result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Dict[str, Any]:
        objects = {}
        split = DatasetSplit[config.get("split").upper()]
        for stage in Stage:
            stage_config = _build_stage_config(config, stage)
            dataset_builder = self.DATASET_BUILDER[split][stage_config.pop("type")]
            processor_config = [TARGET_EXTRACTOR_PROCESSOR_CONFIG] if not stage_config.has_path("processors") \
                else stage_config.pop("processors")
            loader_config = build_default_loader_config(stage_config, stage, [dataset_builder], processor_config)

            objects[stage.value] = self._factory.build(loader_config, context)

        return objects

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ["datasources"]

    def config_key(self) -> str:
        return "datasources"


def _build_stage_config(config: Config, stage: Stage) -> Config:
    overwritten_stage_values = config.get_config([stage.value])
    default_values = _exclude_keys(config, [s.value for s in Stage])
    for key in overwritten_stage_values.get_keys():
        default_values.set(key, overwritten_stage_values.get(key))

    return default_values


def _exclude_keys(config: Config, key_to_exclude: List[str]) -> Config:
    new_config = copy.deepcopy(config)
    for key in key_to_exclude:
        new_config.pop(key)
    return new_config
