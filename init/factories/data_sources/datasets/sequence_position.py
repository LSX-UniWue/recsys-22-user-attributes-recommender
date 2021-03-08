from pathlib import Path
from typing import List

from data.datasets.index import SequencePositionIndex
from data.datasets.sequence_position import SequencePositionDataset
from data.datasets.processors.processor import Processor
from data.datasets.sequence import ItemSessionParser
from init.config import Config
from init.context import Context
from init.factories.data_sources.datasets.plain_session import PlainSessionDatasetFactory
from init.factories.util import require_config_field_equal, require_config_keys
from init.object_factory import CanBuildResult, CanBuildResultType
from init.factories.data_sources.datasets.dataset_factory import DatasetFactory


class SequencePositionDatasetFactory(DatasetFactory):

    REQUIRED_FIELDS = ["nip_index_file"]

    def __init__(self):
        super(SequencePositionDatasetFactory, self).__init__()
        self.plain_session_factory = PlainSessionDatasetFactory()

    def _can_build_dataset(self, config: Config, context: Context) -> CanBuildResult:
        result = require_config_field_equal(config, 'type', 'sequence_position')
        if result.type != CanBuildResultType.CAN_BUILD:
            return result

        result = self.plain_session_factory.can_build(config, context)
        if result.type != CanBuildResultType.CAN_BUILD:
            return result

        return require_config_keys(config, self.REQUIRED_FIELDS)

    def _build_dataset(self,
                       config: Config,
                       context: Context,
                       session_parser: ItemSessionParser,
                       processors: List[Processor]
                       ) -> SequencePositionDataset:
        nip_index_file_path = Path(config.get('nip_index_file'))
        basic_dataset = self.plain_session_factory.build(config, context)

        add_target = config.get_or_default("add_target", True)

        return SequencePositionDataset(basic_dataset,
                                       SequencePositionIndex(nip_index_file_path),
                                       processors=processors,
                                       add_target=add_target)
