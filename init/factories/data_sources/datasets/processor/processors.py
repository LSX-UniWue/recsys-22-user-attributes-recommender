from typing import Any, List

from data.datasets.processors.processor import Processor
from init.config import Config
from init.context import Context
from init.factories.data_sources.datasets.processor.cloze_mask import ClozeProcessorFactory
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ProcessorsFactory(ObjectFactory):
    KEY = 'processors'

    def __init__(self):
        super().__init__()

        # FIXME: register other processors
        self.processors_factories = [ClozeProcessorFactory()]

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> List[Processor]:

        processors = []
        # FIXME: how to select a list item?
        for subconfig in config.get([]):
            subconfig = Config(subconfig)

            for processor_factory in self.processors_factories:
                can_build = processor_factory.can_build(subconfig, context)
                if can_build.type == CanBuildResultType.CAN_BUILD:
                    processor = processor_factory.build(subconfig, context)
                    processors.append(processor)

        return processors

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return ['processors']

    def config_key(self) -> str:
        return self.KEY
