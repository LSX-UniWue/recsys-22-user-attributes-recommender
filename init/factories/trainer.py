from typing import Union, Any, Dict, List

from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from runner.util.builder import TrainerBuilder


class TrainerFactory(ObjectFactory):

    KEY = "trainer"

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    # TODO initialize from config
    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        return TrainerBuilder().build()

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
