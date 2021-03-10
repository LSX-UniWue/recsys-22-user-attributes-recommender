import torch

from metrics.common import calc_ndcg
from metrics.metric import RankingMetric


class NormalizedDiscountedCumulativeGainMetric(RankingMetric):

    """
    calculates the Normalized Discounted Cumulative Gain (NDCG) at k
    """

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("ndcg", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('count', torch.tensor(0), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                positive_item_mask: torch.Tensor,
                metric_mask: torch.Tensor
                ) -> None:
        """

        :param prediction: the logits for I items :math`(N, I)`
        :param positive_item_mask: a mask where a 1 indices that the item at this index is relevant :math`(N, I)`
        :return:
        """
        ndcg = calc_ndcg(prediction, positive_item_mask, self._k, metric_mask)
        self.ndcg += ndcg.sum()
        self.count += prediction.size()[0]

    def compute(self):
        return self.ndcg / self.count

    def name(self):
        return f"NDCG@{self._k}"
