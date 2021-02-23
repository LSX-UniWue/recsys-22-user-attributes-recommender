from typing import List

from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from metrics.metric import RankingMetric


class RankingMetricFactory(ObjectFactory):

    def __init__(self,
                 key: str,
                 ranking_cls):
        super().__init__()
        self.key = key
        self.ranking_cls = ranking_cls

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> List[RankingMetric]:
        ks = config.get([])
        if not isinstance(ks, list):
            ks = [ks]
        return [self.ranking_cls(k) for k in ks]

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return [self.key]

    def config_key(self) -> str:
        return self.key