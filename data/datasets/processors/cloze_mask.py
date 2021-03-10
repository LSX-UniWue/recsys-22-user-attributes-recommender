import random
from typing import Dict, Any, List

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from data.datasets.processors.processor import Processor
from tokenization.tokenizer import Tokenizer

TOKENIZER_PREFIX = 'tokenizers.'
ITEM_TOKENIZER = 'tokenizers.item'


class ClozeMaskProcessor(Processor):
    """
    A processor, that replaces with a given probability items in the sequence
    with a mask token that the model should than predict (e.g. BERT4Rec)

    Example:
        Input:
            session: [1, 5, 7, 8]
        Output:
            session:          [1, 5, 101, 8]
            targets:          [0, 0, 7,   0]

        where 101 is the mask token id
        please use 0 in the target for loss masking

    """

    def __init__(self,
                 tokenizers: Dict[str, Tokenizer],
                 mask_prob: float,
                 only_last_item_mask_prob: float,
                 seed: int,
                 masking_targets: List[str] = [ITEM_SEQ_ENTRY_NAME]
                 ):
        """
        :param tokenizers: the tokenizers
        :param mask_prob: the mask prob to use for masking items in the sequence
        :param only_last_item_mask_prob: the prob that the last item in the sequence should only be masked
        :param seed: the seed to use
        """
        super().__init__()

        self.tokenizers = tokenizers

        self.mask_prob = mask_prob
        self.only_last_item_mask_prob = only_last_item_mask_prob
        self.masking_targets = masking_targets

        self.random = random.Random(seed)

    def process(self,
                parsed_session: Dict[str, Any]
                ) -> Dict[str, Any]:
        session = parsed_session[ITEM_SEQ_ENTRY_NAME]
        target = session.copy()
        sessions = []

        for mask_target in self.masking_targets:
            sessions.append((mask_target, parsed_session[mask_target]))

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

        # first we decide if we only mask the last item
        mask_last_item_prob = self.random.random()
        if mask_last_item_prob <= self.only_last_item_mask_prob:
            for mask_target, session in sessions:
                mask = get_mask(mask_target, session)
                last_item = len(session) - 1
                session[last_item] = mask
                if mask_target in ITEM_SEQ_ENTRY_NAME:
                    target[:last_item] = [mask] * last_item
        else:
            for index in range(0, len(session)):
                prob = self.random.random()
                if prob < self.mask_prob:
                    prob = prob / self.mask_prob

                    if prob < 0.8:
                        for mask_target, session in sessions:
                            session[index] = get_mask(mask_target, session)
                    elif prob < 0.9:
                        for mask_target, session in sessions:
                            random_index = self.random.randint(0, len(get_tokenizer(mask_target)) - 1)
                            session[index] = format_if_list(random_index, session)
                else:
                    # we use the padding token as masking the cross entropy loss
                    target[index] = self.tokenizers['tokenizers.item'].pad_token_id

        parsed_session[TARGET_ENTRY_NAME] = target

        return parsed_session
