import torch

from metrics.ranking.common import calc_precision
from metrics.ranking.ranking_metric import RankingMetric


class PrecisionAtMetric(RankingMetric):

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

    def name(self):
        return f"precision_at_{self._k}"

