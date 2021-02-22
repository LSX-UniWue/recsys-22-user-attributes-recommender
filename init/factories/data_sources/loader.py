from typing import Any, List, Optional, Dict

from torch.utils.data import DataLoader

from data.collate import padded_session_collate, PadDirection
from data.datasets import ITEM_SEQ_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME, \
    TARGET_ENTRY_NAME
from data.mp import mp_worker_init_fn
from init.config import Config
from init.context import Context
from init.factories.common.dependencies_factory import DependenciesFactory
from init.factories.common.union_factory import UnionFactory
from init.factories.data_sources.datasets.item_session import ItemSessionDatasetFactory
from init.factories.data_sources.datasets.next_item import NextItemDatasetFactory
from init.factories.util import check_config_keys_exist, check_context_entries_exists
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class LoaderFactory(ObjectFactory):
    KEY = "loader"

    DATASET_DEPENDENCY_KEY = "dataset"
    REQUIRED_CONFIG_PARAMS = [
        "batch_size",
        "max_seq_length",
        "max_seq_step_length",
        "pad_direction",
        "shuffle",
        "num_workers"
    ]

    #TODO (AD) make used tokenizer a parameter -> KeBert4Rec
    TOKENIZER_CONTEXT_KEY = "tokenizers.item"
    REQUIRED_CONTEXT_ENTRIES = [
        TOKENIZER_CONTEXT_KEY
    ]

    def __init__(self,
                 dependencies: DependenciesFactory = DependenciesFactory(
                     [
                        UnionFactory([
                            ItemSessionDatasetFactory(),
                            NextItemDatasetFactory()
                        ], DATASET_DEPENDENCY_KEY, [DATASET_DEPENDENCY_KEY])
                     ]
                 )):
        super(LoaderFactory, self).__init__()
        self._dependencies = dependencies

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        dependencies_result = self._dependencies.can_build(config, context)
        if dependencies_result.type != CanBuildResultType.CAN_BUILD:
            return dependencies_result

        if not check_config_keys_exist(config, self.REQUIRED_CONFIG_PARAMS):
            return CanBuildResult(
                CanBuildResultType.MISSING_CONFIGURATION,
                f"Could not find all required keys ({self.REQUIRED_CONFIG_PARAMS}) in config."
            )

        if not check_context_entries_exists(context, self.REQUIRED_CONTEXT_ENTRIES):
            return CanBuildResult(
                CanBuildResultType.MISSING_DEPENDENCY,
                f"Could not find one of the dependency within the context."
            )

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    #FIXME (AD) make collate_fn its own factory with dependencies on entries to pad ...
    def build(self, config: Config, context: Context) -> Any:
        dependencies = self._dependencies.build(config, context)

        dataset = dependencies[self.DATASET_DEPENDENCY_KEY]
        # TODO next
        tokenizer = context.get(self.TOKENIZER_CONTEXT_KEY)
        num_workers = config.get_or_default("num_workers", 0)

        init_worker_fn = None if num_workers == 0 else mp_worker_init_fn

        pad_direction = PadDirection.LEFT if config.get("pad_direction") == "left" else PadDirection.RIGHT

        return DataLoader(
            dataset=dataset,
            batch_size=config.get("batch_size"),
            shuffle=config.get("shuffle"),
            num_workers=num_workers,
            worker_init_fn=init_worker_fn,
            collate_fn=padded_session_collate(
                pad_token_id=tokenizer.pad_token_id,
                entries_to_pad=self._build_entries_to_pad(config.get("max_seq_length"), config.get("max_seq_step_length")),
                session_length_entry=ITEM_SEQ_ENTRY_NAME,
                pad_direction=pad_direction
            )
        )

    def _build_entries_to_pad(self, max_seq_length: int,
                              max_seq_step_length: Optional[int]
                              ) -> Dict[str, List[int]]:
        entries_to_pad = {
            ITEM_SEQ_ENTRY_NAME: [max_seq_length],
            POSITIVE_SAMPLES_ENTRY_NAME: [max_seq_length],
            NEGATIVE_SAMPLES_ENTRY_NAME: [max_seq_length],
            TARGET_ENTRY_NAME: [max_seq_length]
        }

        if max_seq_step_length is not None:
            for key in [ITEM_SEQ_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME]:
                entries_to_pad[key].append(max_seq_step_length)
            entries_to_pad[TARGET_ENTRY_NAME] = [max_seq_step_length]

        return entries_to_pad

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY