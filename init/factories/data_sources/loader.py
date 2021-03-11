import multiprocessing
from typing import Any, List, Dict, Union

from torch.utils.data import DataLoader

from data.collate import padded_session_collate, PadDirection, PadInformation
from data.datasets import ITEM_SEQ_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME, \
    TARGET_ENTRY_NAME
from data.mp import mp_worker_init_fn
from init.config import Config
from init.context import Context
from init.factories.common.dependencies_factory import DependenciesFactory
from init.factories.common.union_factory import UnionFactory
from init.factories.data_sources.datasets.item_session import ItemSessionDatasetFactory
from init.factories.tokenizer.tokenizer_factory import get_tokenizer_key_for_voc
from init.factories.data_sources.datasets.sequence_position import SequencePositionDatasetFactory
from init.factories.util import check_config_keys_exist, check_context_entries_exists
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


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

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        dependencies_result = self._dependencies.can_build(config, context)
        if dependencies_result.type != CanBuildResultType.CAN_BUILD:
            return dependencies_result

        if not check_config_keys_exist(config, self.REQUIRED_CONFIG_KEYS):
            return CanBuildResult(
                CanBuildResultType.MISSING_CONFIGURATION,
                f"Could not find all required keys ({self.REQUIRED_CONFIG_KEYS}) in config."
            )

        if not check_context_entries_exists(context, self.REQUIRED_CONTEXT_ENTRIES):
            return CanBuildResult(
                CanBuildResultType.MISSING_DEPENDENCY,
                f"Could not find one of the dependency within the context."
            )

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    # FIXME (AD) make collate_fn its own factory with dependencies on entries to pad ...
    def build(self, config: Config, context: Context) -> Any:
        dependencies = self._dependencies.build(config, context)

        dataset = dependencies[self.DATASET_DEPENDENCY_KEY]

        num_workers = config.get_or_default("num_workers", multiprocessing.cpu_count() - 1)

        init_worker_fn = None if num_workers == 0 else mp_worker_init_fn

        pad_direction = PadDirection.LEFT if config.get("pad_direction") == "left" else PadDirection.RIGHT

        return DataLoader(
            dataset=dataset,
            batch_size=config.get("batch_size"),
            shuffle=config.get("shuffle"),
            num_workers=num_workers,
            worker_init_fn=init_worker_fn,
            collate_fn=padded_session_collate(
                entries_to_pad=self._build_entries_to_pad(config.get("max_seq_length"),
                                                          config.get("max_seq_step_length"), context),
                session_length_entry=ITEM_SEQ_ENTRY_NAME,
                pad_direction=pad_direction
            )
        )

    def _build_entries_to_pad(self,
                              max_seq_length: int,
                              max_seq_step_length: Union[Dict[str, int], int],
                              context: Context
                              ) -> Dict[str, PadInformation]:

        tokenizer = context.get(self.TOKENIZER_CONTEXT_KEY)
        item_padding_token = tokenizer.pad_token_id

        item_padding_information = PadInformation(item_padding_token, max_seq_length)
        entries_to_pad = {
            ITEM_SEQ_ENTRY_NAME: item_padding_information,
            POSITIVE_SAMPLES_ENTRY_NAME: item_padding_information,
            NEGATIVE_SAMPLES_ENTRY_NAME: item_padding_information,
            TARGET_ENTRY_NAME: item_padding_information,
        }

        if max_seq_step_length is not None:
            max_seq_step_length_info = max_seq_step_length

            if isinstance(max_seq_step_length, int):
                max_seq_step_length_info = {"item": max_seq_step_length}
            for key, max_length in max_seq_step_length_info.items():
                if key == "item":
                    for entry_key in [ITEM_SEQ_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME,
                                      TARGET_ENTRY_NAME]:
                        entries_to_pad[entry_key].max_seq_step_length = max_length
                else:
                    padding_token = context.get(get_tokenizer_key_for_voc(key)).pad_token_id
                    entries_to_pad[key] = PadInformation(padding_token, max_seq_length, max_length)

        return entries_to_pad

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
