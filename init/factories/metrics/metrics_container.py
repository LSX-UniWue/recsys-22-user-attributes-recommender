from typing import List, Union, Any, Dict

from init.config import Config
from init.context import Context
from init.factories.common.dependencies_factory import DependenciesFactory
from init.factories.metrics.full import FullMetricsFactory
from init.object_factory import ObjectFactory, CanBuildResult


class MetricsContainerFactory(ObjectFactory):

    def __init__(self):
        super().__init__()

        self.metric_factories = DependenciesFactory([FullMetricsFactory()])

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.metric_factories.can_build(config, context)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        metrics = self.metric_factories.build(config, context)
        pass

    def is_required(self, context: Context) -> bool:
        pass

    def config_path(self) -> List[str]:
        return ['metrics']

    def config_key(self) -> str:
        pass