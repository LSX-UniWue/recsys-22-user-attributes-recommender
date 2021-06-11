from typing import List

from asme.init.factories.data_sources.datasets.processor.last_item_mask import get_all_tokenizers_from_context, \
    get_sequence_feature_names
from data.datasets.processors.cloze_mask import ClozeMaskProcessor
from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc, ITEM_TOKENIZER_ID
from asme.init.factories.util import check_config_keys_exist
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ClozeProcessorFactory(ObjectFactory):
    """
    factory for the ClozeMaskProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        # check for config keys
        config_keys_exist = check_config_keys_exist(config, ['mask_probability', 'only_last_item_mask_prob'])
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        if not context.has_path(get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID)):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> ClozeMaskProcessor:

        tokenizers = get_all_tokenizers_from_context(context)

        mask_probability = config.get('mask_probability')
        only_last_item_mask_prob = config.get('only_last_item_mask_prob')

        masking_targets = get_sequence_feature_names(config, context)

        return ClozeMaskProcessor(tokenizers, mask_probability, only_last_item_mask_prob, masking_targets)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'cloze_processor'
