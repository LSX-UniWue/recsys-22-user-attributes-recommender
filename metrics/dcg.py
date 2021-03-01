import torch

from metrics.common import calc_dcg
from metrics.metric import RankingMetric


class DiscountedCumulativeGainMetric(RankingMetric):

    """
    calculates the Discounted Cumulative Gain (DCG) at k
    """

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
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
        self.dcg += calc_dcg(prediction, target, self._k).sum()
        self.count += target.size()[0]

    def compute(self):
        return self.dcg / self.count

    def name(self):
        return f"DCG@{self._k}"
