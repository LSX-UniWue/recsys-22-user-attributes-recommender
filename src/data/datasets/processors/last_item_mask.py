from typing import Dict, Any, List

from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.datasets.processors.processor import Processor
from asme.tokenization.tokenizer import Tokenizer

TOKENIZER_PREFIX = 'tokenizers.'
ITEM_TOKENIZER = 'tokenizers.item'


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
                 tokenizers: Dict[str, Tokenizer],
                 masking_targets: List[str] = [ITEM_SEQ_ENTRY_NAME]

                 ):
        super().__init__()

        self.tokenizers = tokenizers
        self.masking_targets = masking_targets

    def process(self,
                parsed_sequence: Dict[str, Any]
                ) -> Dict[str, Any]:

        def get_tokenizer(target):
            if target in [ITEM_SEQ_ENTRY_NAME]:
                return self.tokenizers[ITEM_TOKENIZER]
            else:
                return self.tokenizers[TOKENIZER_PREFIX + target]

        def get_mask(target, session):
            tokenizer = get_tokenizer(target)
            return format_if_list(tokenizer.mask_token_id, session)

        def format_if_list(item, session):
            if isinstance(session[0], list):
                return [item]
            return item

        for target in self.masking_targets:
            session = parsed_sequence[target]
            mask_token = get_mask(target, session)
            session.append(mask_token)

        return parsed_sequence
