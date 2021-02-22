from typing import List, Union, Any, Dict

from data.datasets.processors.processor import Processor
from data.datasets.session import ItemSessionParser
from init.config import Config
from init.context import Context
from init.factories.common.dependencies_factory import DependenciesFactory
from init.factories.data_sources.datasets.parser.item_session_parser import ItemSessionParserFactory
from init.factories.data_sources.datasets.processor.processors import ProcessorsFactory
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType

from abc import abstractmethod


class DatasetFactory(ObjectFactory):

    def __init__(self):
        super().__init__()
        self.parser_dependency = DependenciesFactory([ItemSessionParserFactory(), ProcessorsFactory()])

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        can_build = self.parser_dependency.can_build(config, context)
        if can_build.type != CanBuildResultType.CAN_BUILD:
            return can_build

        return self._can_build_dataset(config, context)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        build_result = self.parser_dependency.build(config, context)

        session_parser = build_result[ItemSessionParserFactory.KEY]
        processors = build_result[ProcessorsFactory.KEY]

        return self._build_dataset(config, context, session_parser, processors)

    @abstractmethod
    def _build_dataset(self,
                       config: Config,
                       context: Context,
                       session_parser: ItemSessionParser,
                       processors: List[Processor]
                       ) -> Any:
        pass

    @abstractmethod
    def _can_build_dataset(self,
                           config: Config,
                           context: Context
                           ) -> CanBuildResult:
        pass
