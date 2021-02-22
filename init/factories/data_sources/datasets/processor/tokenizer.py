from typing import List

from data.datasets.processors.last_item_mask import LastItemMaskProcessor
from data.datasets.processors.tokenizer import TokenizerProcessor
from init.config import Config
from init.context import Context
from init.factories.tokenizer.tokenizer_factory import TokenizerFactory
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class TokenizerProcessorFactory(ObjectFactory):

    #FIXME make this configurable for other tokenizers, e.g. keywords
    TOKENIZER_KEY = TokenizerFactory.KEY + '.item'

    """
    Factory for the TokenizerProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        if context.has_path(self.TOKENIZER_KEY):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> TokenizerProcessor:
        tokenizer = context.get(self.TOKENIZER_KEY)

        return TokenizerProcessor(tokenizer)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'tokenizer_processor'
