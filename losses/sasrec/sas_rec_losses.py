import torch
from torch import nn

IGNORE_INDEX = -100
DEFAULT_REDUCTION = 'mean'


class SASRecBinaryCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str = DEFAULT_REDUCTION):
        super().__init__()
        self.reduction = reduction

    def forward(self,
                pos_input: torch.Tensor,
                neg_input: torch.Tensor,
                mask: torch.Tensor):

        return sas_binary_cross_entropy(pos_input, neg_input, mask, self.reduction)


def _log_sigmoid(tensor: torch.Tensor,
                 eps: float = 1e-24,
                 reverse: bool = False) -> torch.Tensor:
    tensor_eps = torch.sigmoid(tensor) + eps
    if reverse:
        tensor_eps = 1 - tensor_eps
    return torch.log(tensor_eps)


def sas_binary_cross_entropy(pos_input: torch.Tensor,
                             neg_input: torch.Tensor,
                             mask: torch.Tensor,
                             reduction: str = DEFAULT_REDUCTION
                             ) -> torch.Tensor:
    """
    :param pos_input: the positive logits
    :param neg_input: the negative logits
    :param mask: the mask to apply (padding)
    :param reduction: the reduction to perform
    :return:
    """

    pos = _log_sigmoid(pos_input) * mask
    neg = _log_sigmoid(neg_input, reverse=True) * mask

    sum = torch.sum(-pos - neg) / torch.sum(mask)
    if reduction == 'mean':
        return torch.mean(sum)
    return sum
