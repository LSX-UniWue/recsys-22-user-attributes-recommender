from typing import Dict, Any

from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.datasets.processors.processor import Processor
from tokenization.tokenizer import Tokenizer


class LastItemMaskProcessor(Processor):

    def __init__(self,
                 tokenizer: Tokenizer,
                 ):
        super().__init__()

        self.tokenizer = tokenizer

    def process(self, parsed_session: Dict[str, Any]) -> Dict[str, Any]:
        session = parsed_session[ITEM_SEQ_ENTRY_NAME]
        # just add a mask token at the end of the sequence
        session.append(self.tokenizer.mask_token_id)
        return parsed_session