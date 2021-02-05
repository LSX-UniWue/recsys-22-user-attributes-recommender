from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor


@dataclass
class MetricsSample:
    sampled_predictions: Tensor
    """
    A tensor containing the predictions of the model for the sampled items.
    """

    positive_item_mask: Tensor
    """
    A mask that marks the positions of the target items in the :code sampled_predictions tensor.
    """


class MetricsSampler:
    """
    Generates negative samples based on the target item space, targets and actual input.

    The sampler draws from the target item distribution as defined by the :code weights provided
    on initialization. Also care is taken that the target items and items in the input sequence do not appear in the
    generated sample.

    After the sample is drawn, the target scores for the target items are added.

    Example:
        If the :code sample_size is :code 100, and there is one target, the final sample has size :code 101
    """

    def __init__(self, weights: List[float], sample_size: int):
        """
        Constructor.

        :param weights: weights for each item in target space.
        :param sample_size: number of samples generated for evaluation.
        """
        self.weights = weights
        self.sample_size = sample_size

    def sample(self,
               input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> MetricsSample:
        """
        Generates :code weights negative samples for each entry in the batch.
        
        :param input_seq: the batch of input sequences. 
        :param targets:  the targets for the provided batch.
        :param predictions: the predictions for the input_seq made by the model.
        :param mask: ??? TODO: add documentation
         
        :return: the negative sample.
        """

        weight = torch.tensor(self.weights, device=predictions.device)
        batch_size = input_seq.size()[0]
        weight = weight.unsqueeze(0).repeat(batch_size, 1)

        # never sample targets
        batch_indices = torch.arange(0, batch_size)
        weight[batch_indices, targets] = 0.

        # we want to use scatter to set the items contained in the input to 0 for every row in the batch
        src = torch.ones_like(input_seq).to(torch.long)
        sequence_mask = torch.zeros_like(weight).to(torch.long)

        # calculate a mask where 1. signals that the item should get 0. probability since it occurs in the input
        # sequence.
        sequence_mask = sequence_mask.scatter(1, input_seq, src)
        weight[sequence_mask.to(dtype=torch.bool)] = 0.

        sampled_negatives = torch.multinomial(weight, num_samples=self.sample_size)
        target_batched = targets.unsqueeze(1)
        sampled_items = torch.cat([target_batched, sampled_negatives], dim=1)

        positive_item_mask = sampled_items.eq(target_batched).to(dtype=predictions.dtype)
        # FIXME: fix positive_item_mask with mask
        sampled_predictions = predictions.gather(1, sampled_items)

        return MetricsSample(sampled_predictions, positive_item_mask)

    def get_sample_size(self) -> int:
        return self.sample_size
