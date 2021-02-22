from typing import List, Union, Any, Dict

from init.config import Config
from init.context import Context
from init.factories.common.dependencies_factory import DependenciesFactory
from init.factories.metrics.full_metrics import FullMetricsFactory
from init.factories.metrics.sampled_metrics import SampledMetricsFactory
from init.object_factory import ObjectFactory, CanBuildResult
from metrics.container.aggregate_metrics_container import AggregateMetricsContainer


class MetricsContainerFactory(ObjectFactory):

    def __init__(self):
        super().__init__()

        self.metric_factories = DependenciesFactory([FullMetricsFactory(), SampledMetricsFactory()],
                                                    optional_based_on_path=True)

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.metric_factories.can_build(config, context)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        metrics_dict = self.metric_factories.build(config, context)
        return AggregateMetricsContainer(list(metrics_dict.values()))

    def is_required(self, context: Context) -> bool:
        pass

    def config_path(self) -> List[str]:
        return ['metrics']

    def config_key(self) -> str:
        pass
