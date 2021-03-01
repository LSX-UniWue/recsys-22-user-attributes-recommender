from typing import List, Any

from init.config import Config
from init.context import Context
from init.factories.common.dependencies_factory import DependenciesFactory
from init.factories.metrics.ranking_metric import RankingMetricFactory
from init.object_factory import ObjectFactory, CanBuildResult
from metrics.dcg import DiscountedCumulativeGainMetric
from metrics.f1 import F1Metric
from metrics.mrr import MRRMetric
from metrics.ndcg import NormalizedDiscountedCumulativeGainMetric
from metrics.precision import PrecisionMetric
from metrics.metric import RankingMetric
from metrics.recall import RecallMetric


def _collect(metrics_dict: [str, List[Any]]) -> List[Any]:
    return [item for sublist in metrics_dict.values() for item in sublist]


class MetricsFactory(ObjectFactory):

    def __init__(self):
        super().__init__()

        # TODO: config this
        self.metrics = DependenciesFactory([RankingMetricFactory('mrr', MRRMetric),
                                            RankingMetricFactory('f1', F1Metric),
                                            RankingMetricFactory('recall', RecallMetric),
                                            RankingMetricFactory('dcg', DiscountedCumulativeGainMetric),
                                            RankingMetricFactory('ndcg', NormalizedDiscountedCumulativeGainMetric),
                                            RankingMetricFactory('precision', PrecisionMetric)],
                                           optional_based_on_path=True)

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.metrics.can_build(config, context)

    def build(self, config: Config, context: Context) -> List[RankingMetric]:
        metrics = self.metrics.build(config, context)
        return _collect(metrics)

    def is_required(self, context: Context) -> bool:
        pass

    def config_path(self) -> List[str]:
        return ['metrics']

    def config_key(self) -> str:
        return 'metrics'
