from typing import List

from asme.core.evaluation.evaluation import LogInputEvaluator, ExtractRecommendationEvaluator, TrueTargetEvaluator
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.data.datasets.processors.no_target_extractor import NoTargetExtractorProcessor


class TrueTargetEvaluatorFactory(ObjectFactory):
    """
    Factory for the TrueTargetEvaluator
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> TrueTargetEvaluator:


        item_tokenizer = context.get("tokenizers.item")

        return TrueTargetEvaluator(item_tokenizer=item_tokenizer)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'evaluation'
