from typing import List

from asme.core.evaluation.evaluation import LogInputEvaluator, ExtractRecommendationEvaluator, ExtractSampleIdEvaluator
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.data.datasets.processors.no_target_extractor import NoTargetExtractorProcessor


class ExtractSampleIdEvaluatorFactory(ObjectFactory):
    """
    Factory for the ExtractSampleIdEvaluator
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> ExtractSampleIdEvaluator:

        use_session_id = config.get_or_default("session_id", True)
        return ExtractSampleIdEvaluator(use_session_id=use_session_id)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'recommendation'
