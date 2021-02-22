from pathlib import Path
from typing import List

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets.index import SessionPositionIndex
from data.datasets.nextitem import NextItemDataset
from data.datasets.processors.processor import Processor
from data.datasets.session import PlainSessionDataset, ItemSessionParser
from init.config import Config
from init.context import Context
from init.factories.util import require_config_field_equal
from init.object_factory import CanBuildResult
from init.factories.data_sources.datasets.dataset_factory import DatasetFactory


class NextItemDatasetFactory(DatasetFactory):

    def __init__(self):
        super(NextItemDatasetFactory, self).__init__()

    def _can_build_dataset(self, config: Config, context: Context) -> CanBuildResult:
        return require_config_field_equal(config, 'type', 'nextit')

    def _build_dataset(self,
                       config: Config,
                       context: Context,
                       session_parser: ItemSessionParser,
                       processors: List[Processor]
                       ) -> NextItemDataset:
        csv_file_path = Path(config.get('csv_file'))
        csv_file_index_path = Path(config.get('csv_file_index'))
        nip_index_file_path = Path(config.get('nip_index_file'))
        index = CsvDatasetIndex(csv_file_index_path)
        reader = CsvDatasetReader(csv_file_path, index)
        basic_dataset = PlainSessionDataset(reader, session_parser)

        return NextItemDataset(basic_dataset, SessionPositionIndex(nip_index_file_path), processors=processors)

    def is_required(self, context: Context) -> bool:
        pass

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        pass
