from typing import List

from data.datasets.processors.tokenizer import TokenizerProcessor
from init.config import Config
from init.context import Context
from init.factories.tokenizer.tokenizer_factory import get_tokenizer_key_for_voc
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class TokenizerProcessorFactory(ObjectFactory):

    """
    Factory for the TokenizerProcessor
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
              ) -> TokenizerProcessor:
        tokenizer = context.get(get_tokenizer_key_for_voc("item"))

        return TokenizerProcessor(tokenizer)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'tokenizer_processor'
