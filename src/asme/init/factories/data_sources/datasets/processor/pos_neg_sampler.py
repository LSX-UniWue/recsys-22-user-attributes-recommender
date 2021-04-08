from typing import List

from data.datasets.processors.pos_neg_sampler import PositiveNegativeSamplerProcessor
from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.tokenizer.tokenizer_factory import get_tokenizer_key_for_voc
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class PositiveNegativeSamplerProcessorFactory(ObjectFactory):

    """
    factory for the PositiveNegativeSamplerProcessor
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
              ) -> PositiveNegativeSamplerProcessor:
        tokenizer = context.get(get_tokenizer_key_for_voc("item"))

        return PositiveNegativeSamplerProcessor(tokenizer)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'pos_neg_sampler_processor'
