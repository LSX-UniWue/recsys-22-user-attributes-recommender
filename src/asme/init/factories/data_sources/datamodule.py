from typing import List, Union, Any, Dict

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.data_sources.data_sources import DataSourcesFactory
from asme.init.factories.util import check_config_keys_exist
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class DataModuleFactory(ObjectFactory):

    CONFIG_KEY = "datamodule"
    REQUIRED_CONFIG_KEYS = ["dataset_name"]

    def __init__(self):
        super().__init__()
        self._datasources_factory = DataSourcesFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        dependencies_result = self._datasources_factory.can_build(config, context)
        if dependencies_result.type != CanBuildResultType.CAN_BUILD:
            return dependencies_result

        if not check_config_keys_exist(config, self.REQUIRED_CONFIG_KEYS):
            return CanBuildResult(
                CanBuildResultType.MISSING_CONFIGURATION,
                f"Could not find all required keys ({self.REQUIRED_CONFIG_KEYS}) in config."
            )

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        pass

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.CONFIG_KEY]

    def config_key(self) -> str:
        return  self.CONFIG_KEY