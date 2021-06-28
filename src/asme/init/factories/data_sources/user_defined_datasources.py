from typing import List, Union, Any, Dict

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.init.templating.datasources.datasources import Stage, DatasetSplit


class UserDefinedDataSourcesFactory(ObjectFactory):

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
        # TODO: Add pop method to config class
        config_dict = config.config
        for stage in Stage:
            prefix = config_dict.pop("file_prefix")
            split = DatasetSplit[config_dict.pop("split").upper()]
            path = config_dict.pop("path")
            objects[stage] = self._factory.build(config.get_config(stage.value), context)

        return objects

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ["datasources"]

    def config_key(self) -> str:
        return "datasources"

    def _infer_properties(self, config: Config, stage: Stage):
        pass
