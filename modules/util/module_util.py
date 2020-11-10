import torch
from torch.nn import functional as F

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


def convert_target_for_multi_label_margin_loss(target: torch.Tensor,
                                               num_classes: int,
                                               pad_token_id: int
                                               ) -> torch.Tensor:
    """
    pads the (already padded) target vector the the number of classes,
    sets every pad token to a negative value (e.g. required for the MultiLabelMarginLoss of PyTorch

    :param target: the target tensor, containing target class ids :math `(N, X)`
    :param num_classes: the number of classes for the multi-class multi-classification
    :param pad_token_id: the padding token id
    :return: a tensor of shape :math `(N, C)`

    where C is the number of classes, N the batch size, and X the padded class id length
    """
    converted_target = F.pad(target, [0, num_classes - target.size()[1]], value=pad_token_id)
    # TODO: also mask all special tokens
    converted_target[converted_target == pad_token_id] = -1
    return converted_target


def convert_target_to_multi_hot(target_tensor: torch.Tensor,
                                num_classes: int,
                                pad_token_id: int
                                ) -> torch.Tensor:
    """
    generates a mulit-hot vector of the provided indices in target_tensor
    :param target_tensor:
    :param num_classes:
    :param pad_token_id:
    :return: a tensor with 1s in the indices specified by
    """
    multi_hot = torch.zeros(target_tensor.size()[0], num_classes, device=target_tensor.device)
    multi_hot.scatter_(1, target_tensor, 1)
    # remove the padding for each multi-hot target
    multi_hot[:, pad_token_id] = 0
    return multi_hot
