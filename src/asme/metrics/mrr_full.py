import torch

from asme.metrics.common import get_true_positives
from asme.metrics.metric import RankingMetric


class MRRFullMetric(RankingMetric):

    """
    calculates the Mean Reciprocal Rank (MRR)
    """

    def __init__(self,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("mrr", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

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
        device = prediction.device


        num_positions = prediction.size()[1] + 1
        tp = get_true_positives(prediction, positive_item_mask, num_positions , metric_mask)
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
        return f"MRR"
