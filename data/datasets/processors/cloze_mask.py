import random
from typing import Dict, Any

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from data.datasets.processors.processor import Processor
from tokenization.tokenizer import Tokenizer


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
                 tokenizer: Tokenizer,
                 mask_prob: float,
                 only_last_item_mask_prob: float,
                 seed: int
                 ):
        """
        :param tokenizer: the tokenizer
        :param mask_prob: the mask prob to use for masking items in the sequence
        :param only_last_item_mask_prob: the prob that the last item in the sequence should only be masked
        :param seed: the seed to use
        """
        super().__init__()

        self.tokenizer = tokenizer

        self.mask_prob = mask_prob
        self.only_last_item_mask_prob = only_last_item_mask_prob

        self.random = random.Random(seed)

    def process(self,
                parsed_session: Dict[str, Any]
                ) -> Dict[str, Any]:
        session = parsed_session[ITEM_SEQ_ENTRY_NAME]
        target = session.copy()

        # first we decide if we only mask the last item
        mask_last_item_prob = self.random.random()
        if mask_last_item_prob <= self.only_last_item_mask_prob:
            last_item = len(session) - 1
            session[last_item] = self.tokenizer.mask_token_id
            target[:last_item] = [self.tokenizer.pad_token_id] * last_item
        else:
            for index in range(0, len(session)):
                prob = self.random.random()
                if prob < self.mask_prob:
                    prob = prob / self.mask_prob

                    if prob < 0.8:
                        session[index] = self.tokenizer.mask_token_id
                    elif prob < 0.9:
                        session[index] = self.random.randint(0, len(self.tokenizer) - 1)
                else:
                    # we use the padding token as masking the cross entropy loss
                    target[index] = self.tokenizer.pad_token_id

        parsed_session[TARGET_ENTRY_NAME] = target
        return parsed_session
