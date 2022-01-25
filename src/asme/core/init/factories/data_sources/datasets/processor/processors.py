from typing import List

from asme.core.init.factories.data_sources.datasets.processor.target_extractor import TargetExtractorProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.no_target_extractor import NoTargetExtractorProcessorFactory
from asme.data.datasets.processors.processor import Processor
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.core.init.factories.common.config_based_factory import ListFactory
from asme.core.init.factories.data_sources.datasets.processor.cloze_mask import ClozeProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.cut_to_fixed_sequence_length import \
    CutToFixedSequenceLengthProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.last_item_mask import LastItemMaskProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.positive_item_extractor import \
    PositiveItemExtractorProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.negative_item_sampler import \
    NegativeItemSamplerProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.pos_neg_sampler import PositiveNegativeSamplerProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.position_token import PositionTokenProcessorFactory
from asme.core.init.factories.data_sources.datasets.processor.tokenizer import TokenizerProcessorFactory
from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.data.datasets.processors.registry import REGISTERED_PREPROCESSORS

FIXED_SEQUENCE_LENGTH_PROCESSOR_KEY = 'fixed_sequence_length_processor'


class ProcessorsFactory(ObjectFactory):
    KEY = 'processors'

    def __init__(self):
        super().__init__()

        # FIXME: register other processors
        self.processors_factories = ListFactory(
            ConditionalFactory(
                'type',
                REGISTERED_PREPROCESSORS
            )
        )

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
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
