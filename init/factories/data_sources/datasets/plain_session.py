from typing import List

from data.datasets.session import PlainSessionDataset
from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult


class PlainSessionDatasetFactory(ObjectFactory):

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        pass

    def build(self, config: Config, context: Context) -> PlainSessionDataset:
        pass

    def is_required(self, context: Context) -> bool:
        pass

    def config_path(self) -> List[str]:
        pass

    def config_key(self) -> str:
        pass
