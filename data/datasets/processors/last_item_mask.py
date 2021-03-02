from typing import Dict, Any

from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.datasets.processors.processor import Processor
from tokenization.tokenizer import Tokenizer


class LastItemMaskProcessor(Processor):

    """
    Adds a mask token at the end of the input sequence.
    This is useful for evaluation purposes in some models, e.g. BERT4Rec.

    Example:
        Input:
            session: [1, 5, 7, 8]
        Output:
            session:          [1, 5, 7, 8, 101]

    where 101 is the mask token id
    """

    def __init__(self,
                 tokenizer: Tokenizer,
                 ):
        super().__init__()

        self.tokenizer = tokenizer

    def process(self, parsed_session: Dict[str, Any]) -> Dict[str, Any]:
        session = parsed_session[ITEM_SEQ_ENTRY_NAME]
        # just add a mask token at the end of the sequence
        mask_token = self.tokenizer.mask_token_id

        # check for basket recommendation
        # TODO: maybe config another processor?
        if isinstance(session[0], list):
            mask_token = [mask_token]
        session.append(mask_token)
        return parsed_session
