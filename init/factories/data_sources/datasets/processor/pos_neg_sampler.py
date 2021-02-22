from typing import List

from data.datasets.processors.pos_neg_sampler import PositiveNegativeSamplerProcessor
from init.config import Config
from init.context import Context
from init.factories.util import require_config_field_equal
from init.object_factory import ObjectFactory, CanBuildResult


class PositiveNegativeSamplerProcessorFactory(ObjectFactory):

    """
    factory for the PositiveNegativeSamplerProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return require_config_field_equal(config, 'type', 'pos_neg_sampler')

    def build(self,
              config: Config,
              context: Context
              ) -> PositiveNegativeSamplerProcessor:
        tokenizer = context.get('tokenizers.item')
        seed = config.get('seed')

        return PositiveNegativeSamplerProcessor(tokenizer, seed)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'pos_neg_sampler_processor'
