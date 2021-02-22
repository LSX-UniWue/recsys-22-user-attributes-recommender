from typing import List

from data.datasets.processors.cloze_mask import ClozeMaskProcessor
from init.config import Config
from init.context import Context
from init.factories.util import require_config_field_equal
from init.object_factory import ObjectFactory, CanBuildResult


class ClozeProcessorFactory(ObjectFactory):

    """
    factory for the ClozeMaskProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return require_config_field_equal(config, 'type', 'cloze')

    def build(self,
              config: Config,
              context: Context
              ) -> ClozeMaskProcessor:
        tokenizer = context.get('tokenizers.item')
        mask_probability = config.get('mask_probability')
        only_last_item_mask_prob = config.get('only_last_item_mask_prob')
        seed = config.get('seed')

        return ClozeMaskProcessor(tokenizer, mask_probability, only_last_item_mask_prob, seed)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'cloze_processor'
