from typing import List

from data.datasets.processors.last_item_mask import LastItemMaskProcessor
from init.config import Config
from init.context import Context
from init.factories.tokenizer.tokenizer_factory import TokenizerFactory, get_tokenizer_key_for_voc
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class LastItemMaskProcessorFactory(ObjectFactory):
    #FIXME make this configurable for other tokenizers, e.g. keywords
    TOKENIZER_KEY = TokenizerFactory.KEY + '.item'

    """
    Factory for the LastItemMaskProcessor.
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        if not context.has_path(get_tokenizer_key_for_voc("item")):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> LastItemMaskProcessor:
        tokenizer = context.get(get_tokenizer_key_for_voc("item"))

        return LastItemMaskProcessor(tokenizer)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'pos_neg_sampler_processor'
