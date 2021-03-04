from typing import List

from data.datasets.processors.cloze_mask import ClozeMaskProcessor
from init.config import Config
from init.context import Context
from init.factories.tokenizer.tokenizer_factory import TokenizerFactory, get_tokenizer_key_for_voc
from init.factories.util import check_config_keys_exist
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ClozeProcessorFactory(ObjectFactory):

    """
    factory for the ClozeMaskProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        # check for config keys
        config_keys_exist = check_config_keys_exist(config, ['mask_probability', 'only_last_item_mask_prob', 'seed'])
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        if not context.has_path(get_tokenizer_key_for_voc("item")):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> ClozeMaskProcessor:
        tokenizer = context.get(get_tokenizer_key_for_voc("item"))
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
