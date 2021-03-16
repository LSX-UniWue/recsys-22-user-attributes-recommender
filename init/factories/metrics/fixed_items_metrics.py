from pathlib import Path
from typing import List

from init.config import Config
from init.context import Context
from init.factories.metrics.metrics import MetricsFactory
from init.factories.util import require_config_keys
from init.object_factory import ObjectFactory, CanBuildResult
from metrics.container.metrics_sampler import FixedItemsSampler
from metrics.container.metrics_container import RankingMetricsContainer
from utils.ioutils import load_file_with_item_ids


class FixedItemsMetricsFactory(ObjectFactory):

    """
    a factory to build the fixed items metrics container
    """

    KEY = 'fixed'

    def __init__(self):
        super().__init__()
        self.metrics_factory = MetricsFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return require_config_keys(config, ['metrics', 'item_file'])

    def build(self, config: Config, context: Context) -> RankingMetricsContainer:
        metrics = self.metrics_factory.build(config.get_config(self.metrics_factory.config_path()), context)
        items = load_file_with_item_ids(Path(config.get('item_file')))

        sampler = FixedItemsSampler(items)
        return RankingMetricsContainer(metrics, sampler)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
