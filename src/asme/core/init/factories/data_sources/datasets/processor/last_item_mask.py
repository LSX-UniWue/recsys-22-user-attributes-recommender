from typing import List

from asme.core.init.factories.util import get_all_tokenizers_from_context
from asme.core.init.factories import BuildContext
from asme.data.datasets.processors.last_item_mask import LastItemMaskProcessor
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc, ITEM_TOKENIZER_ID
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.data.datasets.sequence import MetaInformation


# TODO (DZO): move constants and methods
CONTEXT_META_INFO_KEY = "metadata_info"


def get_meta_infos(config: Config,
                   context: Context) -> List[MetaInformation]:
    return context.get("features")


def get_sequence_feature_names(config: Config,
                               context: Context
                               ) -> List[str]:
    return [info.feature_name for info in get_sequence_features(config, context)]


def get_sequence_features(config: Config,
                          context: Context
                          ) -> List[MetaInformation]:
    """
    returns all sequence attributes (the sequence and all sequence meta data configured)
    :param config: the config where we are currently extracting the information from
    :param context: the context to load the meta data information from
    :return: a list of targets to mask
    """
    meta_infos = get_meta_infos(config, context)

    return [info for info in meta_infos if info.is_sequence]


def get_meta_data_path(config: Config) -> List[str]:
    """ returns the key to use for storing the meta data information in the context """
    return config.base_path[:2] + [CONTEXT_META_INFO_KEY]


class LastItemMaskProcessorFactory(ObjectFactory):

    """
    Factory for the LastItemMaskProcessor.
    """

    def can_build(self,
                  build_context: BuildContext
                  ) -> CanBuildResult:
        if not build_context.get_context().has_path(get_tokenizer_key_for_voc(ITEM_TOKENIZER_ID)):
            return CanBuildResult(CanBuildResultType.MISSING_DEPENDENCY, 'item tokenizer missing')

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              build_context: BuildContext
              ) -> LastItemMaskProcessor:
        config = build_context.get_current_config_section()
        context = build_context.get_context()

        tokenizers = get_all_tokenizers_from_context(context)

        masking_targets = get_sequence_feature_names(config, context)
        return LastItemMaskProcessor(tokenizers, masking_targets)

    def is_required(self, build_context: BuildContext) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'last_item_processor'
