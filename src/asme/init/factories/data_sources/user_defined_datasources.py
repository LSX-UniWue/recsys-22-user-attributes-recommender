from typing import List, Union, Any, Dict

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.init.templating.datasources.datasources import Stage, DatasetSplit, NextPositionDatasetBuilder, \
    SequenceDatasetRatioSplitBuilder, NextPositionWindowDatasetBuilder, LeaveOneOutSessionDatasetBuilder, \
    LeaveOneOutSequenceWindowDatasetBuilder, LeaveOneOutNextPositionDatasetBuilder


class UserDefinedDataSourcesFactory(ObjectFactory):
    DATASET_BUILDER = {
        DatasetSplit.RATIO_SPLIT: {
            "next_item": NextPositionDatasetBuilder(),
            "sequence": SequenceDatasetRatioSplitBuilder(),
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
        self._factory = ConditionalFactory("type", {

        })

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        for stage in Stage:
            result = self._factory.can_build(config.get_config(stage.value), context)
            if result.type != CanBuildResultType.CAN_BUILD:
                return result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Dict[str, Any]:
        objects = {}
        split = DatasetSplit[config.pop("split").upper()]
        for stage in Stage:
            stage_config = config.get_config(stage.value)
            generated_config = self.DATASET_BUILDER[split][stage_config.pop("type")]\
                .build_dataset_definition(stage, config)
            dataset_config = Config(generated_config)

            # Overwrite keys that were inferred with the values given in the config
            for k in stage_config.get_keys():
                v = stage_config.get(k)
                dataset_config.set(k, v)

            objects[stage] = self._factory.build(stage_config, context)

        return objects

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ["datasources"]

    def config_key(self) -> str:
        return "datasources"

    def _infer_properties(self, config: Config, stage: Stage):
        pass
