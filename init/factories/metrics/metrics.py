from typing import List, Any

from init.config import Config
from init.context import Context
from init.factories.common.dependencies_factory import DependenciesFactory
from init.factories.metrics.ranking_metric import RankingMetricFactory
from init.object_factory import ObjectFactory, CanBuildResult
from metrics.ranking.dcg import DiscountedCumulativeGain
from metrics.ranking.f1_at import F1AtMetric
from metrics.ranking.mrr_at import MRRAtMetric
from metrics.ranking.ndcg import NormalizedDiscountedCumulativeGain
from metrics.ranking.precision_at import PrecisionAtMetric
from metrics.ranking.ranking_metric import RankingMetric
from metrics.ranking.recall_at import RecallAtMetric


def _collect(metrics_dict: [str, List[Any]]) -> List[Any]:
    return [item for sublist in metrics_dict.values() for item in sublist]


class MetricsFactory(ObjectFactory):

    def __init__(self):
        super().__init__()

        #TODO: config this
        self.metrics = DependenciesFactory([RankingMetricFactory('mrr', MRRAtMetric),
                                            RankingMetricFactory('f1', F1AtMetric),
                                            RankingMetricFactory('recall', RecallAtMetric),
                                            RankingMetricFactory('dcg', DiscountedCumulativeGain),
                                            RankingMetricFactory('ndcg', NormalizedDiscountedCumulativeGain),
                                            RankingMetricFactory('precision', PrecisionAtMetric)],
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
