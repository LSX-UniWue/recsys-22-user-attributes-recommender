import torch
from pytorch_lightning.metrics.functional.reduction import reduce
from torch import nn

DEFAULT_REDUCTION = 'elementwise_mean'


class SASRecBinaryCrossEntropyLoss(nn.Module):
    """
    The adapted binary cross entropy loss from the SASRec paper
    """

    def __init__(self, reduction: str = DEFAULT_REDUCTION):
        """
        inits the binary cross entropy loss of the SASRec paper
        :param reduction: the reduction to use
        """
        super().__init__()
        self.reduction = reduction

    def forward(self,
                pos_input: torch.Tensor,
                neg_input: torch.Tensor,
                mask: torch.Tensor):
        """
        calculates the adapted binary cross entropy of the given positive and negative logits
        :param pos_input: the positive logits
        :param neg_input: the negative logits
        :param mask: the mask to apply (padding)
        :return: the sas rec binary cross entropy
        """

        return sas_rec_binary_cross_entropy(pos_input, neg_input, mask, self.reduction)


def _log_sigmoid(tensor: torch.Tensor,
                 eps: float = 1e-24,
                 reverse: bool = False) -> torch.Tensor:
    tensor_eps = torch.sigmoid(tensor) + eps
    if reverse:
        tensor_eps = 1 - tensor_eps
    return torch.log(tensor_eps)


def sas_rec_binary_cross_entropy(pos_input: torch.Tensor,
                                 neg_input: torch.Tensor,
                                 mask: torch.Tensor,
                                 reduction: str = DEFAULT_REDUCTION
                                 ) -> torch.Tensor:
    """
    :param pos_input: the positive logits
    :param neg_input: the negative logits
    :param mask: the mask to apply (padding)
    :param reduction: the reduction to perform
    :return: the SASRec binary corss entropy for the given inputs
    """
    # AD: for some reason the transformer takes masks where 'True' signals the presence of a pad token. For this code
    # to work, we need to negate the mask first

    mask = ~mask
    pos = _log_sigmoid(pos_input) * mask
    neg = _log_sigmoid(neg_input, reverse=True) * mask

    avg_sum = torch.sum(- pos - neg) / torch.sum(mask)
    return reduce(avg_sum, reduction)
