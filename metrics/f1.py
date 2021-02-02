import torch

from metrics.common import calc_precision, calc_recall
from metrics.metric import RankingMetric


class F1Metric(RankingMetric):

    """
    calculates the F1 score at k
    """

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("f1", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('count', torch.tensor(0.), dist_reduce_fx="sum")

    def _update(self,
                predictions: torch.Tensor,
                positive_item_mask: torch.Tensor
                ) -> None:
        precision = calc_precision(predictions, positive_item_mask, self._k)
        recall = calc_recall(predictions, positive_item_mask, self._k)

        # the f1 is the harmonic mean of recall and precision
        f1 = 2 * recall * precision / (recall + precision)
        f1[torch.isnan(f1)] = 0.0

        self.f1 += precision.sum()
        self.count += predictions.size()[0]

    def compute(self):
        return self.precision / self.count

    def name(self):
        return f"f1@{self._k}"
