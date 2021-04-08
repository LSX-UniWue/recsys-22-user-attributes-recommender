import random
from typing import Dict, Any, List, Union

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from data.datasets.processors.processor import Processor
from asme.tokenization.tokenizer import Tokenizer

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
        sequence = parsed_session[ITEM_SEQ_ENTRY_NAME]
        target = sequence.copy()
        sequences = {
            mask_target: parsed_session[mask_target] for mask_target in self.masking_targets
        }

        basket_recommendation = isinstance(sequence[0], list)

        def _format_item(token_id: int) -> Union[int, List[int]]:
            return [token_id] if basket_recommendation else token_id

        def get_tokenizer(target):
            if target in [ITEM_SEQ_ENTRY_NAME]:
                return self.tokenizers[ITEM_TOKENIZER]

            return self.tokenizers[TOKENIZER_PREFIX + target]

        # first we decide if we only mask the last item
        mask_last_item_prob = self.random.random()
        if mask_last_item_prob <= self.only_last_item_mask_prob:

            for mask_target, sequence in sequences.items():
                tokenizer = get_tokenizer(mask_target)

                last_item = len(sequence) - 1
                sequence[last_item] = _format_item(tokenizer.mask_token_id)

                # if it is the original sequence, update the target
                if mask_target == ITEM_SEQ_ENTRY_NAME:
                    padding_mask = tokenizer.pad_token_id
                    target[:last_item] = [padding_mask] * last_item
        else:
            for index in range(0, len(sequence)):
                prob = self.random.random()
                if prob < self.mask_prob:
                    prob = prob / self.mask_prob

                    if prob < 0.8:
                        for mask_target, sequence in sequence:
                            sequence[index] = _format_item(get_tokenizer(mask_target).mask_token_id)
                    elif prob < 0.9:
                        for mask_target, sequence in sequences:
                            random_index = self.random.randint(0, len(get_tokenizer(mask_target)) - 1)
                            sequence[index] = _format_item(random_index)
                else:
                    # we use the padding token as masking the cross entropy loss
                    target[index] = self.tokenizers['tokenizers.item'].pad_token_id

        parsed_session[TARGET_ENTRY_NAME] = target

        return parsed_session
