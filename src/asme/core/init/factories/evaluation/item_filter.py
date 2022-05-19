from typing import List

from asme.core.evaluation.item_filter import FilterPredictionItems
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class FilterPredictionItemsFactory(ObjectFactory):
    """
    Factory for the ExtractRecommendationEvaluator
    """
    KEY = "filter_items"

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> FilterPredictionItems:

        filter_file = config.get_or_default("file", None)
        return FilterPredictionItems(selected_items_file=filter_file)
    
    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
