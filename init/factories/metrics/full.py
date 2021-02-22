from typing import List

from init.config import Config
from init.context import Context
from init.factories.metrics.metrics import MetricsFactory
from init.factories.util import require_config_keys
from init.object_factory import ObjectFactory, CanBuildResult
from metrics.container.ranking_metrics_container import RankingMetricsContainer


class FullMetricsFactory(ObjectFactory):
    KEY = 'full'

    def __init__(self):
        super().__init__()
        self.metrics_factory = MetricsFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return require_config_keys(config, ['metrics']) and self.metrics_factory.can_build(config, context)

    def build(self, config: Config, context: Context) -> RankingMetricsContainer:
        metrics = self.metrics_factory.build(config.get_config(self.metrics_factory.config_path()), context)
        return RankingMetricsContainer(metrics)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
