from typing import Any, List

from data.datasets.processors.processor import Processor
from data.datasets.sequence import ItemSessionParser, ItemSequenceDataset
from init.config import Config
from init.context import Context
from init.factories.data_sources.datasets.plain_session import PlainSessionDatasetFactory
from init.factories.util import require_config_field_equal
from init.object_factory import CanBuildResult, CanBuildResultType
from init.factories.data_sources.datasets.dataset_factory import DatasetFactory


class ItemSessionDatasetFactory(DatasetFactory):

    def __init__(self):
        super(ItemSessionDatasetFactory, self).__init__()
        self.plain_session_factory = PlainSessionDatasetFactory()

    def _can_build_dataset(self, config: Config, context: Context) -> CanBuildResult:
        result = require_config_field_equal(config, 'type', 'session')
        if result.type != CanBuildResultType.CAN_BUILD:
            return result

        return self.plain_session_factory.can_build(config, context)

    def _build_dataset(self,
                       config: Config,
                       context: Context,
                       session_parser: ItemSessionParser,
                       processors: List[Processor]) -> Any:
        basic_dataset = self.plain_session_factory.build(config, context)

        return ItemSequenceDataset(basic_dataset, processors=processors)
