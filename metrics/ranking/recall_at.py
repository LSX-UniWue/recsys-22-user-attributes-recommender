import torch

from metrics.ranking.common import calc_recall
from metrics.ranking.ranking_metric import RankingMetric


class RecallAtMetric(RankingMetric):
    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super(RecallAtMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k
        self.add_state("recall", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                targets: torch.Tensor,
                mask: torch.Tensor = None,
                ) -> None:
        """
        :param prediction: the scores of all items :math `(N, I)`
        :param targets: the target label tensor :math `(N, T)` or :math `(N)`
        :param mask: the mask to apply, iff no mask is provided all targets are used for calculating the metric
        :math `(N, I)`
        """

        recall = calc_recall(prediction, targets, self._k, mask=mask)
        self.recall += recall.sum()
        self.count += recall.size()[0]

    def compute(self):
        return self.recall / self.count

    def name(self):
        return f"recall_at_{self._k}"
