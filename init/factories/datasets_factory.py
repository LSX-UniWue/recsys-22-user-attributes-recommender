from typing import List, Any

from init.config import Config
from init.context import Context
from init.factories.dependencies import DependenciesTrait
from init.factories.multiple_elements_factory import MultipleElementsFactoryTrait
from init.factories.select_from_factory import SelectFromFactory
from init.object_factory import ObjectFactory, CanBuildResult


class DataSourcesFactory(ObjectFactory, MultipleElementsFactoryTrait):

    KEY = "data_sources"

    def __init__(self):
        super(DataSourcesFactory, self).__init__()
        self.data_source_factory = DataSourceFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.can_build_elements(config, context, self.data_source_factory)

    def build(self, config: Config, context: Context) -> Any:
        return self.build_elements(config, context, self.data_source_factory)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY

class DataSourceFactory(ObjectFactory):
    KEY = "data_source"

class LoaderFactory(ObjectFactory, DependenciesTrait):
    KEY = "loader"

    def __init__(self):
        super(LoaderFactory, self).__init__()
        self.add_dependency(
            SelectFromFactory(
                "dataset",
                True,
                [
                    ItemSessionDatasetFactory(),
                    NextItemDatasetFactory()
                ]
            )
        )
        self.add_dependency(DataLoaderFactory())

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        self.can_build_dependencies(config, context)

    def build(self, config: Config, context: Context) -> Any:
        pass

    def is_required(self, context: Context) -> bool:
        pass

    def config_path(self) -> List[str]:
        pass

    def config_key(self) -> str:
        pass


class PlainSessionDatasetFactory(ObjectFactory):

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


class ItemSessionDatasetFactory(ObjectFactory, DependenciesTrait):
    def __init__(self):
        super(ItemSessionDatasetFactory, self).__init__()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        pass

    def build(self, config: Config, context: Context) -> Any:
        pass

    def is_required(self, context: Context) -> bool:
        pass

    def config_path(self) -> List[str]:
        pass


class NextItemDatasetFactory(ObjectFactory, DependenciesTrait):
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


class ItemSessionParserFactory(ObjectFactory):

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


class ProcessorsFactory(ObjectFactory):
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


#TODO add specific Factories for every processor
class ClozeProcessorFactory(ObjectFactory):

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


class DataLoaderFactory(ObjectFactory):
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