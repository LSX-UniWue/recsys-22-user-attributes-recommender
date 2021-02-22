from typing import List

from data.datasets.processors.processor import Processor
from init.config import Config
from init.context import Context
from init.factories.common.config_based_factory import ConfigBasedFactory
from init.factories.data_sources.datasets.processor.cloze_mask import ClozeProcessorFactory
from init.factories.data_sources.datasets.processor.pos_neg_sampler import PositiveNegativeSamplerProcessorFactory
from init.object_factory import ObjectFactory, CanBuildResult


class ProcessorsFactory(ObjectFactory):
    KEY = 'processors'

    def __init__(self):
        super().__init__()

        # FIXME: register other processors
        self.processors_factories = ConfigBasedFactory('type', {'cloze': ClozeProcessorFactory(),
                                                                'pos_neg': PositiveNegativeSamplerProcessorFactory()})

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.processors_factories.can_build(config, context)

    def build(self,
              config: Config,
              context: Context
              ) -> List[Processor]:
        return self.processors_factories.build(config, context)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['processors']

    def config_key(self) -> str:
        return self.KEY
