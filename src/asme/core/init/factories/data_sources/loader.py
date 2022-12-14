import multiprocessing
from typing import Any, List, Dict

from asme.core.init.factories import BuildContext
from asme.data.datasets.sequence import MetaInformation
from torch.utils.data import DataLoader

from asme.core.init.factories.data_sources.datasets.processor.last_item_mask import get_sequence_features
from asme.data.collate import padded_session_collate, PadDirection, PadInformation
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME, TARGET_SUFFIX
from asme.data.multi_processing import mp_worker_init_fn
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.common.union_factory import UnionFactory
from asme.core.init.factories.data_sources.datasets.item_session import ItemSessionDatasetFactory
from asme.core.init.factories.features.tokenizer_factory import get_tokenizer_key_for_voc
from asme.core.init.factories.data_sources.datasets.sequence_position import SequencePositionDatasetFactory
from asme.core.init.factories.util import check_config_keys_exist, check_context_entries_exists, \
    can_build_with_subsection, build_with_subsection
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


def _get_item_feature_info(context: Context
                           ) -> MetaInformation:
    feature_infos = context.get('features')
    for feature_info in feature_infos:
        if feature_info.feature_name == 'item':
            return feature_info
    raise ValueError('no item feature info found')


def _build_padding_info(context: Context,
                        meta_data: MetaInformation
                        ) -> PadInformation:
    key = meta_data.feature_name
    padding_token = context.get(get_tokenizer_key_for_voc(key)).pad_token_id
    max_seq_length = meta_data.sequence_length
    max_seq_step_length = meta_data.configs.get('max_sequence_step_length')
    return PadInformation(padding_token, max_seq_length, max_seq_step_length)


def _build_padding_info_map(context: Context,
                            meta_data: MetaInformation
                            ) -> Dict[str, PadInformation]:
    if not meta_data.is_sequence:
        return {}
    key = meta_data.feature_name
    pad_info = _build_padding_info(context, meta_data)
    return {
        key: pad_info,
        key + TARGET_SUFFIX: pad_info,
        NEGATIVE_SAMPLES_ENTRY_NAME+"."+key: pad_info,
        POSITIVE_SAMPLES_ENTRY_NAME+"."+key: pad_info
    }


def _build_entries_to_pad(config: Config,
                          context: Context
                          ) -> Dict[str, PadInformation]:
    entries_to_pad = {}
    # also get the sequence meta data
    sequence_features_info = get_sequence_features(config, context)
    for sequence_feature_info in sequence_features_info:
        padding_infos = _build_padding_info_map(context, sequence_feature_info)
        entries_to_pad.update(padding_infos)

    # add special features generated by the processors
    # therefore we need the feature information of the item
    item_info = _get_item_feature_info(context)
    item_padding_information = _build_padding_info(context, item_info)
    entries_to_pad[POSITIVE_SAMPLES_ENTRY_NAME] = item_padding_information
    entries_to_pad[NEGATIVE_SAMPLES_ENTRY_NAME] = item_padding_information

    return entries_to_pad


class LoaderFactory(ObjectFactory):
    KEY = "loader"

    DATASET_DEPENDENCY_KEY = "dataset"
    REQUIRED_CONFIG_KEYS = [
        "batch_size",
        "max_seq_length",
        "max_seq_step_length",
        "pad_direction",
        "shuffle",
        "num_workers"
    ]

    TOKENIZER_CONTEXT_KEY = "tokenizers.item"
    REQUIRED_CONTEXT_ENTRIES = [
        TOKENIZER_CONTEXT_KEY
    ]

    def __init__(self,
                 dependencies: DependenciesFactory = DependenciesFactory(
                     [
                        UnionFactory([
                            ItemSessionDatasetFactory(),
                            SequencePositionDatasetFactory()
                        ], DATASET_DEPENDENCY_KEY, [DATASET_DEPENDENCY_KEY])
                     ]
                 )):
        super(LoaderFactory, self).__init__()
        self._dependencies = dependencies

    def can_build(self, build_context: BuildContext) -> CanBuildResult:
        dependencies_result = can_build_with_subsection(self._dependencies, build_context)
        if dependencies_result.type != CanBuildResultType.CAN_BUILD:
            return dependencies_result

        if not check_config_keys_exist(build_context.get_current_config_section(), self.REQUIRED_CONFIG_KEYS):
            return CanBuildResult(
                CanBuildResultType.MISSING_CONFIGURATION,
                f"Could not find all required keys ({self.REQUIRED_CONFIG_KEYS}) in config."
            )

        if not check_context_entries_exists(build_context.get_context(), self.REQUIRED_CONTEXT_ENTRIES):
            return CanBuildResult(
                CanBuildResultType.MISSING_DEPENDENCY,
                f"Could not find one of the dependency within the context."
            )

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    # FIXME (AD) make collate_fn its own factory with dependencies on entries to pad ...
    def build(self, build_context: BuildContext) -> Any:
        config = build_context.get_current_config_section()

        dependencies = build_with_subsection(self._dependencies, build_context)

        dataset = dependencies[self.DATASET_DEPENDENCY_KEY]

        num_workers = config.get_or_default("num_workers", multiprocessing.cpu_count() - 1)
        persistent_workers = True if num_workers > 1 else False
        init_worker_fn = None if num_workers == 0 else mp_worker_init_fn

        pad_direction = PadDirection.LEFT if config.get("pad_direction") == "left" else PadDirection.RIGHT
        dynamic_padding = config.get_or_default('dynamic_padding', True)

        shuffle_dataset = config.get("shuffle")
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("batch_size"),
            shuffle=shuffle_dataset,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            worker_init_fn=init_worker_fn,
            collate_fn=padded_session_collate(
                entries_to_pad=_build_entries_to_pad(config, build_context.get_context()),
                session_length_entry=ITEM_SEQ_ENTRY_NAME,
                pad_direction=pad_direction,
                dynamic_padding=dynamic_padding
            )
        )

    def is_required(self, build_context: BuildContext) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
