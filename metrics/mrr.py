import torch

from metrics.common import get_true_positives
from metrics.metric import RankingMetric


class MRRMetric(RankingMetric):

    """
    calculates the Mean Reciprocal Rank (MRR) at k
    """

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("mrr", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                positive_item_mask: torch.Tensor
                ) -> None:
        """
        :param prediction: the logits for I items :math`(N, I)`
        :param positive_item_mask: a mask where a 1 indices that the item at this index is relevant :math`(N, I)`
        :return:
        """
        device = prediction.device

        tp = get_true_positives(prediction, positive_item_mask, self._k)

        num_positions = min(prediction.size()[1], self._k) + 1
        ranks = torch.arange(1, num_positions).unsqueeze(0).repeat(prediction.size()[0], 1).to(device=device)
        rank = (ranks * tp).max(dim=-1).values

        # mrr will contain 'inf' values if target is not in top k scores -> setting it to 0
        mrr = 1 / rank
        mrr[mrr == float('inf')] = 0

        self.mrr += mrr.sum()
        self.count += mrr.size()[0]

    def compute(self):
        return self.mrr / self.count

    def name(self):
        return f"MRR@{self._k}"
