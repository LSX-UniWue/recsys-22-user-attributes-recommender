from pathlib import Path
from typing import Any, List

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets.processors.processor import Processor
from data.datasets.session import ItemSessionParser, PlainSessionDataset, ItemSessionDataset
from init.config import Config
from init.context import Context
from init.factories.util import require_config_field_equal
from init.object_factory import  CanBuildResult
from init.factories.data_sources.datasets.dataset_factory import DatasetFactory


class ItemSessionDatasetFactory(DatasetFactory):

    def __init__(self):
        super(ItemSessionDatasetFactory, self).__init__()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return require_config_field_equal(config, 'type', 'session')

    def _can_build_dataset(self, config: Config, context: Context) -> CanBuildResult:
        pass

    def _build_dataset(self, config: Config, context: Context, session_parser: ItemSessionParser,
                       processors: List[Processor]) -> Any:
        csv_file_path = Path(config.get('csv_file'))
        csv_file_index_path = Path(config.get('csv_file_index'))
        index = CsvDatasetIndex(Path(csv_file_index_path))
        reader = CsvDatasetReader(csv_file_path, index)
        basic_dataset = PlainSessionDataset(reader, session_parser)

        # FIXME: handle bert session reader
        #if truncated_index_path is not None:
        #    index = SessionPositionIndex(Path(truncated_index_path))
        #    return NextItemDataset(basic_dataset,
        #                           index=index,
        #                           processors=processors,
        #                           add_target=False)

        return ItemSessionDataset(basic_dataset, processors=processors)

    def is_required(self, context: Context) -> bool:
        pass

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        pass
