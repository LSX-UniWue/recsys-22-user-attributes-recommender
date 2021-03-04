from pathlib import Path
from typing import Any, List

from data.datasets.sequence import ItemSessionParser
from data.utils import create_indexed_header, read_csv_header
from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ItemSessionParserFactory(ObjectFactory):

    KEY = 'session_parser'

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        # TODO: config validation?
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Any:
        csv_file = Path(config.get('csv_file'))
        parser_config = config.get_config(['parser'])
        delimiter = parser_config.get_or_default('delimiter', '\t')
        item_column_name = parser_config.get('item_column_name')
        item_separator = parser_config.get('item_separator')

        # FIXME: load additional features
        additional_features = {}

        header = create_indexed_header(read_csv_header(csv_file, delimiter=delimiter))
        return ItemSessionParser(header, item_column_name,
                                 additional_features=additional_features,
                                 item_separator=item_separator,
                                 delimiter=delimiter)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return self.KEY
