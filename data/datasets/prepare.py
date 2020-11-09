from abc import abstractmethod
from typing import Dict, Any, List

from data.datasets import ITEM_SEQ_ENTRY_NAME
from tokenization.tokenizer import Tokenizer


class Preprocessor:

    @abstractmethod
    def preprocess(self,
                   parsed_session: Dict[str, Any]) -> Dict[str, Any]:
        pass


def build_processors(processors_config: Dict[str, Any],
                     **kwargs: Dict[str, Any]
                     ) -> List[Preprocessor]:

    processors = []
    for key, config in processors_config.items():
        complete_args = {**kwargs, **config}
        preprocessor = build_processor(key, **complete_args)
        processors.append(preprocessor)
    return processors


def build_processor(processor_id: str,
                    **kwargs
                    ) -> Preprocessor:
    if processor_id == 'tokenizer_processor':
        return TokenizerPreprocessor(kwargs.get('tokenizer'))

    raise NotImplementedError(f"unknown preprocessor {processor_id}")


class TokenizerPreprocessor(Preprocessor):

    def __init__(self,
                 tokenizer: Tokenizer
                 ):
        super().__init__()
        self._tokenizer = tokenizer

    def preprocess(self, parsed_session: Dict[str, Any]) -> Dict[str, Any]:
        items = parsed_session[ITEM_SEQ_ENTRY_NAME]
        tokenized_items = self._tokenizer.convert_tokens_to_ids(items)
        parsed_session[ITEM_SEQ_ENTRY_NAME] = tokenized_items

        return parsed_session
