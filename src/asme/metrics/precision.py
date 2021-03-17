import torch

from asme.metrics.common import calc_precision
from asme.metrics.metric import RankingMetric


class PrecisionMetric(RankingMetric):

    """
    calculates the precision at k
    """

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("precision", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('count', torch.tensor(0), dist_reduce_fx="sum")

    def _update(self,
                predictions: torch.Tensor,
                positive_item_mask: torch.Tensor,
                metric_mask: torch.Tensor
                ) -> None:
        precision = calc_precision(predictions, positive_item_mask, self._k, metric_mask)

        self.precision += precision.sum()
        self.count += predictions.size()[0]

    def compute(self):
        return self.precision / self.count

    def name(self):
        return f"precision@{self._k}"
