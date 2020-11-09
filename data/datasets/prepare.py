from abc import abstractmethod
from typing import Dict, Any, List

from data.datasets import ITEM_SEQ_ENTRY_NAME, POSITION_IDS, TARGET_ENTRY_NAME
from tokenization.tokenizer import Tokenizer


class Processor:

    @abstractmethod
    def process(self,
                parsed_session: Dict[str, Any]) -> Dict[str, Any]:
        pass


def build_processors(processors_config: Dict[str, Any],
                     **kwargs: Dict[str, Any]
                     ) -> List[Processor]:

    processors = []
    for key, config in processors_config.items():
        complete_args = {**kwargs, **config}
        preprocessor = build_processor(key, **complete_args)
        processors.append(preprocessor)
    return processors


def build_processor(processor_id: str,
                    **kwargs
                    ) -> Processor:
    if processor_id == 'tokenizer_processor':
        return TokenizerPreprocessor(kwargs.get('tokenizer'))

    if processor_id == 'position_processor':
        return PositionPreprocessor(kwargs.get('seq_length'))

    raise KeyError(f"unknown preprocessor {processor_id}")


class TokenizerPreprocessor(Processor):

    KEYS_TO_TOKENIZE = [ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME]

    def __init__(self,
                 tokenizer: Tokenizer
                 ):
        super().__init__()
        self._tokenizer = tokenizer

    def process(self, parsed_session: Dict[str, Any]) -> Dict[str, Any]:
        for key in TokenizerPreprocessor.KEYS_TO_TOKENIZE:
            if key in parsed_session:
                items = parsed_session[key]
                tokenized_items = self._tokenizer.convert_tokens_to_ids(items)
                parsed_session[key] = tokenized_items

        return parsed_session


class PositionPreprocessor(Processor):

    def __init__(self,
                 seq_length: int
                 ):
        super().__init__()

        self._seq_length = seq_length

    def _generate_position_tokens(self,
                                  items: List[int]
                                  ) -> List[int]:

        counts = list(map(len, items))

        positions = []
        last_position = 0
        for position, count in enumerate(counts):
            total_count = [position] * count
            positions.extend(total_count)
            last_position += len(total_count)
        # maybe to many items
        positions = positions[0: self._seq_length]
        # fill up the last positions
        end = last_position + self._seq_length - len(positions)
        positions.extend(range(last_position, end))

        assert len(positions) == self._seq_length
        return positions

    def process(self, parsed_session: Dict[str, Any]) -> Dict[str, Any]:
        items = parsed_session[ITEM_SEQ_ENTRY_NAME]
        if not isinstance(items[0], list):
            raise ValueError('seqencse items are not list of lists')

        # generate the positions
        positions = self._generate_position_tokens(items)

        flat_items = [item for sublist in items for item in sublist]
        parsed_session[ITEM_SEQ_ENTRY_NAME] = flat_items
        parsed_session[POSITION_IDS] = positions

        return parsed_session
