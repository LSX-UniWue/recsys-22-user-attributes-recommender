from typing import Dict, Any

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from data.datasets.processors.processor import Processor
from tokenization.tokenizer import Tokenizer


class TokenizerProcessor(Processor):

    """
    Tokenizes the configured fields with the tokenizer

    Example:
        Input:
            session: [item 1, item 2, item 5, item 22]
        Output:
            session:          [1, 5, 7, 8]

        where 1, 5, 7, 8 are the token ids of the corresponding items in the session

    """

    KEYS_TO_TOKENIZE = [ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME]

    def __init__(self,
                 tokenizer: Tokenizer
                 ):
        """
        :param tokenizer: the tokenizer to use for the tokenization
        """
        super().__init__()
        self._tokenizer = tokenizer

    def process(self,
                parsed_session: Dict[str, Any]
                ) -> Dict[str, Any]:
        for key in TokenizerProcessor.KEYS_TO_TOKENIZE:
            if key in parsed_session:
                items = parsed_session[key]
                tokenized_items = self._tokenizer.convert_tokens_to_ids(items)
                parsed_session[key] = tokenized_items

        return parsed_session
