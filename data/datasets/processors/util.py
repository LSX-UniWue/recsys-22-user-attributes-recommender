from typing import Dict, Any, List

from data.datasets.processors.processor import Processor
from data.datasets.processors.cloze_mask import ClozeMaskProcessor
from data.datasets.processors.last_item_mask import LastItemMaskProcessor
from data.datasets.processors.pos_neg_sampler import PositiveNegativeSamplerProcessor
from data.datasets.processors.position_token import PositionTokenProcessor
from data.datasets.processors.tokenizer import TokenizerProcessor

#FIXME (AD): this code is only concerned with initialization and should be moved to a proper location

def build_processors(processors_config: Dict[str, Any],
                     **kwargs: Dict[str, Any]
                     ) -> List[Processor]:

    processors = []

    for processor_config in processors_config:
        for key, config in processor_config.items():
            complete_args = {**kwargs, **config}
            preprocessor = build_processor(key, **complete_args)
            processors.append(preprocessor)
    return processors


def build_processor(processor_id: str,
                    **kwargs
                    ) -> Processor:
    if processor_id == 'tokenizer_processor':
        return TokenizerProcessor(kwargs.get('tokenizer'))

    if processor_id == 'position_processor':
        return PositionTokenProcessor(kwargs.get('seq_length'))

    if processor_id == 'pos_neg_sampler':
        return PositiveNegativeSamplerProcessor(kwargs.get('tokenizer'), seed=kwargs.get('seed'))

    if processor_id == 'mask_processor':
        return ClozeMaskProcessor(kwargs.get('tokenizer'), mask_prob=kwargs.get('mask_prob'),
                                  only_last_item_mask_prob=kwargs.get('last_item_mask_prob'),
                                  seed=kwargs.get('seed'))

    if processor_id == 'mask_eval_processor':
        return LastItemMaskProcessor(kwargs.get('tokenizer'))

    raise KeyError(f"unknown preprocessor {processor_id}")
