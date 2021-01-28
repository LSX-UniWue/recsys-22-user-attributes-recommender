import torch

from metrics.ranking.common import calc_recall, calc_precision
from metrics.ranking.ranking_metric import RankingMetric


class F1AtMetric(RankingMetric):

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

    def name(self):
        return f"f1_at_{self._k}"
