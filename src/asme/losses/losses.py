from abc import abstractmethod

import torch
from torch import nn


DEFAULT_REDUCTION = 'elementwise_mean'


class RecommenderSequenceContrastiveLoss(nn.Module):

    """ A common interface for contrastive losses that use the difference of positive
    and negative samples (e.g. BPR) """

    def __init__(self, reduction: str = DEFAULT_REDUCTION):
        super().__init__()
        self.reduction = reduction

    @abstractmethod
    def forward(self,
                positive_logits: torch.Tensor,
                negative_logits: torch.Tensor,
                mask: torch.Tensor
                ) -> torch.Tensor:
        """
        calculates the loss of the given positive and negative logits
        :param positive_logits: the positive logits :math `(N, S)`
        :param negative_logits: the negative logits :math `(N, S)`
        :param mask: the mask to apply (e.g. for padding) :math `(N, S)`
        true if the position i should be considered for the loss
        :return: the loss

        where N is batch size and S the max sequence length
        """
        pass
