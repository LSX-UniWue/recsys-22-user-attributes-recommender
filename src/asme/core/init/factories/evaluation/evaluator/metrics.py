from typing import List

from asme.core.evaluation.evaluation import PerSampleMetricsEvaluator
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class PerSampleMetricsEvaluatorFactory(ObjectFactory):
    """
    Factory for the ExtractRecommendationEvaluator
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> PerSampleMetricsEvaluator:
        item_tokenizer = context.get("tokenizers.item")
        filter_items = context.get("evaluation")["filter_items"].get_filter()
        module = context.get("module")

        return PerSampleMetricsEvaluator(item_tokenizer=item_tokenizer, filter=filter_items,
                                         module=module)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'evaluation'
