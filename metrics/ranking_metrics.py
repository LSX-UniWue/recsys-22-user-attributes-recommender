from abc import abstractmethod

import pytorch_lightning as pl
import torch


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

        rank = torch.topk((sorted_indices.eq(target) * torch.arange(1, self._k + 1, dtype=torch.float, device=target.device)), k=1).values

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
        tp = tp.sum(1) # get the total number of positive items

        all_relevant_items = positive_item_mask.sum(1)
        recall = tp / all_relevant_items
        self.recall += recall.sum()
        self.count += prediction.size()[0]

    def compute(self):
        return self.recall / self.count
