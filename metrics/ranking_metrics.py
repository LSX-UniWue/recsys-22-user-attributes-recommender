from abc import abstractmethod

import pytorch_lightning as pl
import torch


def reverse_cumsum(input: torch.Tensor,
                   dim: int
                   ) -> torch.Tensor:
    """
    :param input: Tensor to calculate the right to left cumulative sum on
    :param dim: Dimension to use for cumulative sum calculation
    :return: A tensor of same shape as input containing the right to left cumulative sum
    """
    return torch.flip(torch.cumsum(torch.flip(input, dims=[dim]), dim=dim), dims=[dim])


def intersection(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    :param x: Tensor of shape (N,D)
    :param y: Tensor of shape (N,I)
    :return: A Tensor of shape (N,D) containing a 1 at index (i,j) iff x[i][j] is in y[i] and a zero otherwise.
    """
    hits = []
    for i, sample in enumerate(x):
        curr_hits = []
        for val in sample:
            curr_hits.append(1 if val in y[i] else 0)
        hits.append(curr_hits)
    return torch.tensor(data=hits, device=x.device)


def get_tp(predictions: torch.Tensor,
           target: torch.Tensor,
           mask: torch.Tensor,
           k: int
           ) -> torch.Tensor:
    sorted_indices = torch.topk(predictions, k=k).indices
    sorted_indices = torch.repeat_interleave(sorted_indices, mask.size()[-1], dim=0)

    target_expanded = target.view(-1, 1).expand(-1, k)
    tp_mask = torch.repeat_interleave(mask.view(-1, 1), k, dim=1)

    tp = (sorted_indices.eq(target_expanded) * tp_mask).sum(dim=-1).to(dtype=torch.float)
    return tp.view(mask.size()).sum(-1)


def calc_precision(prediction: torch.Tensor,
                   target: torch.Tensor,
                   k: int,
                   mask: torch.Tensor = None
                   ) -> torch.Tensor:
    if len(target.size()) == 1:
        # single dimension, unsqueeze it
        target = torch.unsqueeze(target, dim=1)

    if mask is None:
        mask = torch.ones(target.size(), device=target.device)

    tp = get_tp(predictions=prediction, target=target, mask=mask, k=k)

    return tp / k


def calc_recall(prediction: torch.Tensor,
                target: torch.Tensor,
                k: int,
                mask: torch.Tensor = None
                ) -> torch.Tensor:
    if len(target.size()) == 1:
        # single dimension, unsqueeze it
        target = torch.unsqueeze(target, dim=1)

    if mask is None:
        mask = torch.ones(target.size(), device=target.device)

    tp = get_tp(predictions=prediction, target=target, mask=mask, k=k)
    fn = mask.sum(-1) - tp

    return tp / (tp + fn)


def calc_dcg(prediction: torch.Tensor,
             target: torch.Tensor,
             k: int,
             mask: torch.Tensor = None
             ) -> torch.Tensor:
    """
    Calculates DCG according to https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    :param prediction:
    :param target:
    :param k:
    :param mask:
    :return:
    """

    # ensure target is two-dimensional for the following computations to work properly
    if len(target.size()) == 1:
        target = torch.unsqueeze(target, dim=1)

    # remove masked items if a mask was provided
    masked_target = torch.where(mask > 0, target, -torch.ones_like(target)) if mask is not None else target

    # find the indices of the k items the model believes are the most relevant
    topk = torch.argsort(prediction, descending=True)[:, :k]
    # find the indices of the elements actually matching the targets
    matches = intersection(topk, masked_target)

    # build every possible i till the k position
    positions = torch.arange(2, k + 2, dtype=prediction.dtype)
    # calc every 1 / log2(i+1) and than mask it with the matches
    position_values = 1 / torch.log2(positions) * matches
    return position_values.sum(dim=1)


class RecommendationMetric(pl.metrics.Metric):

    def update(self,
               predictions: torch.Tensor,
               target: torch.Tensor,
               mask: torch.Tensor
               ) -> None:
        self._update(predictions, target, mask)

    @abstractmethod
    def _update(self,
                predictions: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor
                ) -> None:
        pass


class F1AtMetric(RecommendationMetric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False
                 ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("f1", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self._k = k

    def _update(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor = None,
                ) -> None:
        recall = calc_recall(prediction, target, self._k, mask=mask)
        precision = calc_precision(prediction, target, self._k, mask=mask)

        f1 = 2 * recall * precision / (recall + precision)
        f1[torch.isnan(f1)] = 0.0

        self.f1 += f1.sum()
        self.count += f1.size()[0]

    def compute(self):
        return self.f1 / self.count


class PrecisionAtMetric(RecommendationMetric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False
                 ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k
        self.add_state("precision", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor = None,
                ) -> None:
        """
        :param prediction: the scores of all items :math `(N, I)`
        :param target: the target label tensor :math `(N, T)` or :math `(N)`
        :param mask: the mask to apply, iff no mask is provided all targets are used for calculating the metric
        :math `(N, I)`
        """
        precision = calc_precision(prediction, target, self._k, mask=mask)
        self.precision += precision.sum()
        self.count += precision.size()[0]

    def compute(self):
        return self.precision / self.count


class RecallAtMetric(RecommendationMetric):
    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super(RecallAtMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k
        self.add_state("recall", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor = None,
                ) -> None:
        """
        :param prediction: the scores of all items :math `(N, I)`
        :param target: the target label tensor :math `(N, T)` or :math `(N)`
        :param mask: the mask to apply, iff no mask is provided all targets are used for calculating the metric
        :math `(N, I)`
        """

        recall = calc_recall(prediction, target, self._k, mask=mask)
        self.recall += recall.sum()
        self.count += recall.size()[0]

    def compute(self):
        return self.recall / self.count


class MRRAtMetric(RecommendationMetric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super(MRRAtMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("mrr", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor
                ) -> None:
        """
        :param prediction: the scores of all items, the first one is the positive item, all others are negatives :math `(N, I)`
        :param target: the target label tensor :math `(N, T)`
        :return: the mrr at specified k
        """
        sorted_indices = torch.topk(prediction, k=self._k).indices
        target = target.view(-1, 1).expand(-1, self._k)

        rank = torch.topk(
            (sorted_indices.eq(target) * torch.arange(1, self._k + 1, dtype=torch.float, device=target.device)),
            k=1).values

        # mrr will contain 'inf' values if target is not in top k scores -> setting it to 0
        mrr = 1 / rank
        mrr[mrr == float('inf')] = 0

        self.mrr += mrr.sum()
        self.count += mrr.size()[0]

    def compute(self):
        return self.mrr / self.count


class RecommendationSampleMetric(pl.metrics.Metric):

    def update(self,
               predictions: torch.Tensor,
               positive_item_mask: torch.Tensor
               ) -> None:
        self._update(predictions, positive_item_mask)

    @abstractmethod
    def _update(self,
                predictions: torch.Tensor,
                positive_item_mask: torch.Tensor
                ) -> None:
        pass


class RecallAtNegativeSamples(RecommendationSampleMetric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k
        self._num_negative_samples = 100

        self.add_state("recall", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('count', torch.tensor(0.), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                positive_item_mask: torch.Tensor
                ) -> None:
        """

        :param prediction: the logits for I items :math`(N, I)`
        :param positive_item_mask: a mask where a 1 indices that the item at this index is relevant :math`(N, I)`
        :return:
        """
        # get the indices of the top k predictions
        predicted_id = torch.argsort(prediction, descending=True)
        # limit the prediction to the top k
        predicted_id = predicted_id[:, :self._k]

        # select the mask for each of the indices, this is than the tp
        tp = positive_item_mask.gather(1, predicted_id)
        tp = tp.sum(1)  # get the total number of positive items

        all_relevant_items = positive_item_mask.sum(1)
        recall = tp / all_relevant_items
        self.recall += recall.sum()
        self.count += prediction.size()[0]

    def compute(self):
        return self.recall / self.count


class DiscountedCumulativeGain(RecommendationMetric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super(DiscountedCumulativeGain, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("dcg", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('count', torch.tensor(0.), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor = None
                ) -> None:
        """
        :param prediction: the scores of all items, the first one is the positive item, all others are negatives :math `(N, I)`
        :param target: the target label tensor :math `(N, T)`
        :return: DCG@k
        """
        self.dcg += calc_dcg(prediction, target, self._k, mask).sum()
        self.count += target.size()[0]

    def compute(self):
        return self.dcg / self.count


class NormalizedDiscountedCumulativeGain(RecommendationMetric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super(NormalizedDiscountedCumulativeGain, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("ndcg", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('count', torch.tensor(0.), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor = None
                ) -> None:
        """
        :param prediction: the scores of all items, the first one is the positive item, all others are negatives :math `(N, I)`
        :param target: the target label tensor :math `(N, T)`
        """
        dcg = calc_dcg(prediction, target, self._k, mask)

        # Number of items that are expected to be predicted by the model
        target_size = target.size()
        batch_size = target_size[0]

        # Calculate the number of relevant items per sample. If a mask is provided, we only consider the unmasked items
        if mask is not None:
            number_of_relevant_items = mask.sum(dim=1)
        else:
            # If there is only a single dimension, we are in single-item prediction mode
            if len(target_size) == 1:
                number_of_relevant_items = torch.ones((batch_size,))
            # If there is more than one dimension but no mask, all items in the target tensor are considered
            else:
                number_of_relevant_items = torch.full((batch_size,), target_size[1])

        # The model can only suggest k items, therefore the ideal DCG should be calculated using k as an upper bound
        # on the number of relevant items to ensure comparability
        number_of_relevant_items = torch.clamp(number_of_relevant_items, max=self._k)

        # FIXME: The following code seems unnecessarily complicated, is there a nicer and more concise way to achieve the same result?
        number_of_relevant_items = torch.unsqueeze(number_of_relevant_items, dim=1)
        max_number_of_relevant_items = max(number_of_relevant_items.long()).item()
        # This lets us use relevant_positions as an index in the following
        relevant_positions = torch.cat([torch.arange(0, batch_size).unsqueeze(dim=1) ,number_of_relevant_items - 1],
                                       dim=1).long()

        positions_values = torch.zeros((batch_size, max_number_of_relevant_items))
        # This just places 1s at the locations indicated by relevant_positions
        positions_values[relevant_positions[:, 0], relevant_positions[:, 1]] = 1
        # At this point, position_values is filled with as may ones per sample as there are relevant items.
        # The remaining elements per sample are 0s.
        positions_values = reverse_cumsum(positions_values, dim=1)

        # Calculate the ideal DCG with the values calculated above.
        idcg = torch.sum(1 / torch.log2(positions_values * torch.arange(2, max_number_of_relevant_items + 2)), dim=1)

        # Update cumulative ndcg and measurements count
        self.ndcg += (dcg / idcg).sum()
        self.count += dcg.size()[0]

    def compute(self):
        return self.ndcg / self.count
