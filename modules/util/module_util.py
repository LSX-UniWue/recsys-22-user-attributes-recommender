import torch

from tokenization.tokenizer import Tokenizer


def get_padding_mask(tensor: torch.Tensor,
                     tokenizer: Tokenizer,
                     transposed: bool = True,
                     inverse: bool = False) -> torch.Tensor:
    """
    generates the padding mask based on the tokenizer (by default batch first)
    :param tensor:
    :param tokenizer:
    :param transposed:
    :param inverse

    :return:
    """

    if len(tensor.size()) > 2:
        tensor = tensor.max(dim=2).values

    # the masking should be true where the padding token is set
    if inverse:
        padding_mask = tensor.ne(tokenizer.pad_token_id)
    else:
        padding_mask = tensor.eq(tokenizer.pad_token_id)

    if transposed:
        return padding_mask.transpose(0, 1)

    return padding_mask