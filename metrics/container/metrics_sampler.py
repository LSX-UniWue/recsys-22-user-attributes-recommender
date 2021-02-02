from abc import abstractmethod
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
    abstract class to implement to sample only a subset of items for evaluation
    """

    @abstractmethod
    def sample(self,
               input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> MetricsSample:
        pass

    @abstractmethod
    def suffix_metric_name(self) -> str:
        pass


class AllItemsSampler(MetricsSampler):

    """
        A sampler that samples all items
    """

    def sample(self,
               input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> MetricsSample:
        positive_item_mask = _to_multi_hot(predictions.size(), targets)
        return MetricsSample(predictions, positive_item_mask)

    def suffix_metric_name(self) -> str:
        return ""


def _to_multi_hot(size: List[int],
                  targets: torch.Tensor
                  ) -> torch.Tensor:
    multihot = torch.zeros(size).to(torch.long)
    src = torch.ones_like(multihot).to(torch.long)
    if len(targets.size()) == 1:
        targets = targets.unsqueeze(1)
    return multihot.scatter(1, targets, src)


class FixedItemsSampler(MetricsSampler):
    """
    this sampler always returns only the configured items
    """

    def __init__(self, fixed_items: List[int]):
        super().__init__()
        self.fixed_items = fixed_items

    def sample(self,
               input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> MetricsSample:
        fixed_items = torch.LongTensor(self.fixed_items).to(targets.device)
        fixed_items = fixed_items.unsqueeze(0).repeat(targets.size()[0], 1)

        sampled_predictions = predictions.gather(1, fixed_items)

        if len(targets.size()) == 1:
            # next-item recommendation
            target_batched = targets.unsqueeze(1)
            positive_item_mask = fixed_items.eq(target_batched).to(dtype=predictions.dtype)
        else:
            # here we multi-hot encode the targets (1 for each target in the item space)
            multihot = _to_multi_hot(predictions.size(), targets)
            # and than gather the corresponding indices by the fixed items
            positive_item_mask = multihot.gather(1, fixed_items)

        return MetricsSample(sampled_predictions, positive_item_mask)

    def suffix_metric_name(self) -> str:
        return "/fixed_sampled"


class NegativeMetricsSampler(MetricsSampler):
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
        super().__init__()
        self.weights = weights
        self.sample_size = sample_size

    def sample(self,
               input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None
               ) -> MetricsSample:
        """
        Generates :code weights negative samples for each entry in the batch.
        
        :param input_seq: the batch of input sequences. 
        :param targets:  the targets for the provided batch.
        :param predictions: the predictions for the input_seq made by the model.
        :param mask: ???
         
        :return: the negative sample.
        """

        weight = torch.tensor(self.weights, device=predictions.device)
        weight = weight.unsqueeze(0).repeat(input_seq.size()[0], 1)

        # never sample targets
        weight[:, targets] = 0.

        # we want to use scatter to set the items contained in the input to 0 for every row in the batch
        src = torch.ones_like(input_seq).to(torch.long)
        mask = torch.zeros_like(weight).to(torch.long)

        # calculate a mask where 1. signals that the item should get 0. probability since it occurs in the input
        # sequence.
        mask.scatter_(1, input_seq, src)
        weight[mask.to(dtype=torch.bool)] = 0.

        sampled_negatives = torch.multinomial(weight, num_samples=self.sample_size)
        target_batched = targets.unsqueeze(1)
        sampled_items = torch.cat([target_batched, sampled_negatives], dim=1)

        positive_item_mask = sampled_items.eq(target_batched).to(dtype=predictions.dtype)
        # FIXME: fix positive_item_mask with mask
        sampled_predictions = predictions.gather(1, sampled_items)

        return MetricsSample(sampled_predictions, positive_item_mask)

    def suffix_metric_name(self) -> str:
        return f"/sampled({self.sample_size})"
