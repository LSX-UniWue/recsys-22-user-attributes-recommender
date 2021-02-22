from typing import Any, List

from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult


class NextItemDatasetFactory(ObjectFactory):
    def __init__(self):
        super(NextItemDatasetFactory, self).__init__()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        pass

    def build(self, config: Config, context: Context) -> Any:
        pass

    def is_required(self, context: Context) -> bool:
        pass

    def config_path(self) -> List[str]:
        pass

    def config_key(self) -> str:
        pass