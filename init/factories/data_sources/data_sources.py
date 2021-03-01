from typing import Any, List

from init.config import Config
from init.context import Context
from init.factories.common.list_elements_factory import NamedListElementsFactory
from init.factories.data_sources.loader import LoaderFactory
from init.object_factory import ObjectFactory, CanBuildResult


class DataSourcesFactory(ObjectFactory):

    KEY = "data_sources"

    def __init__(self, elements_factory: NamedListElementsFactory = NamedListElementsFactory(LoaderFactory())):
        super(DataSourcesFactory, self).__init__()
        self.elements_factory = elements_factory

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.elements_factory.can_build(config, context)

    def build(self, config: Config, context: Context) -> Any:
        return self.elements_factory.build(config, context)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
