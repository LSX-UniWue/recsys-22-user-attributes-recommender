from typing import List

from data.datasets.processors.tokenizer import TokenizerProcessor
from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.tokenizer.tokenizer_factory import get_tokenizer_key_for_voc
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from data.datasets.processors.processor import DelegatingProcessor, Processor


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
              ) -> Processor:

        tokenizers = context.as_dict()

        tokenizer_processors = []
        for name, tokenizer in tokenizers.items():
            if name.startswith("tokenizers."):
                keys_to_tokenize = name.replace("tokenizers.","")
                if keys_to_tokenize == "item":
                    tokenizer_processors.append(TokenizerProcessor(tokenizer))
                else:
                    tokenizer_processors.append(TokenizerProcessor(tokenizer, [keys_to_tokenize]))

        return DelegatingProcessor(tokenizer_processors)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'tokenizer_processor'
