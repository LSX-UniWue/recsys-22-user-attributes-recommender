from typing import List

from data.datasets.processors.pos_neg_sampler import PositiveNegativeSamplerProcessor
from init.config import Config
from init.context import Context
from init.factories.tokenizer.tokenizer_factory import TokenizerFactory
from init.factories.util import check_config_keys_exist
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class PositiveNegativeSamplerProcessorFactory(ObjectFactory):

    TOKENIZER_KEY = TokenizerFactory.KEY + '.item'

    """
    factory for the PositiveNegativeSamplerProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        config_keys_exist = check_config_keys_exist(config, ['seed'])
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        if context.has_path(self.TOKENIZER_KEY):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> PositiveNegativeSamplerProcessor:
        tokenizer = context.get(self.TOKENIZER_KEY)
        seed = config.get('seed')

        return PositiveNegativeSamplerProcessor(tokenizer, seed)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'pos_neg_sampler_processor'
