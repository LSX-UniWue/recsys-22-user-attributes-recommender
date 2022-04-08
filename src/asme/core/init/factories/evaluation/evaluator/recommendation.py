from typing import List

from asme.core.evaluation.evaluation import LogInputEvaluator, ExtractRecommendationEvaluator
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.data.datasets.processors.no_target_extractor import NoTargetExtractorProcessor


class ExtractRecommendationEvaluatorFactory(ObjectFactory):
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
              ) -> ExtractRecommendationEvaluator:


        item_tokenizer = context.get("tokenizers.item")
        filter_items = context.get("evaluation")["filter_items"].get_filter()
        selected_items = context.get("evaluation")["filter_items"].get_selected_items()
        num_predictions = config.get("number_predictions")

        return ExtractRecommendationEvaluator(item_tokenizer=item_tokenizer, filter=filter_items,
                                              num_predictions=num_predictions, selected_items=selected_items)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'evaluation'
