from typing import Dict, Union, List

import torch
from asme.core.init.factories.features.tokenizer_factory import ITEM_TOKENIZER_ID, get_tokenizer_key_for_voc
from asme.core.tokenization.tokenizer import Tokenizer
from asme.data.datasets import ITEM_SEQ_ENTRY_NAME


def get_mask_token(tokenizers: Dict[str, Tokenizer],
                   target: str,
                   sequence: Union[List[int], List[List[int]]]
                   ) -> Union[int, List[int]]:
    tokenizer = get_tokenizer(tokenizers, target)
    mask_token = tokenizer.mask_token_id

    return [mask_token] if isinstance(sequence[0], list) else mask_token


def get_tokenizer(tokenizers: Dict[str, Tokenizer],
                  target: str
                  ) -> Tokenizer:
    tokenizer_id = target
    if target == ITEM_SEQ_ENTRY_NAME:
        tokenizer_id = ITEM_TOKENIZER_ID

    return tokenizers[get_tokenizer_key_for_voc(tokenizer_id)]


def random_uniform(start: float = 0., end: float = 1.) -> float:
    """
    Draws a single random number uniformly from a continuous distribution (pytorch) in [start; end).

    :param start: lowest number
    :param end: highest number

    :return: a single float from [start; end).
    """
    return torch.empty((), dtype=torch.float, device="cpu").uniform_(start, end).item()


def random_(start: int, end: int) -> int:
    """
    Draws uniformly from a discrete distribution in [start; end]

    :param start: lowest number.
    :param end: highest number.

    :return: a single number.
    """
    return torch.empty((), dtype=torch.int, device="cpu").random_(start, end).item()
