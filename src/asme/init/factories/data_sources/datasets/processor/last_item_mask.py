from typing import List

from data.datasets.processors.last_item_mask import LastItemMaskProcessor
from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.tokenizer.tokenizer_factory import get_tokenizer_key_for_voc, ITEM_TOKENIZER_ID
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class LastItemMaskProcessorFactory(ObjectFactory):

    """
    Factory for the LastItemMaskProcessor.
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        if not context.has_path(get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID)):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> LastItemMaskProcessor:
        tokenizers = context.as_dict()
        masking_targets = config.get("masking_targets")

        return LastItemMaskProcessor(tokenizers, masking_targets)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'last_item_processor'
