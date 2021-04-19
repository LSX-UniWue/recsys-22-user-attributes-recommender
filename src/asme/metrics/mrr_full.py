import torch

from asme.metrics.common import get_true_positives
from asme.metrics.metric import RankingMetric, MetricStorageMode


class MRRFullMetric(RankingMetric):

    """
    calculates the Mean Reciprocal Rank (MRR)
    """

    def __init__(self,
                 dist_sync_on_step: bool = False,
                 storage_mode: MetricStorageMode = MetricStorageMode.SUM):
        super().__init__(metric_id='mrr_full',
                         dist_sync_on_step=dist_sync_on_step,
                         storage_mode=storage_mode)

    def _calc_metric(self,
                     prediction: torch.Tensor,
                     positive_item_mask: torch.Tensor,
                     metric_mask: torch.Tensor
                     ) -> torch.Tensor:
        device = prediction.device

        num_positions = prediction.size()[1] + 1
        tp = get_true_positives(prediction, positive_item_mask, num_positions , metric_mask)
        ranks = torch.arange(1, num_positions).unsqueeze(0).repeat(prediction.size()[0], 1).to(device=device)
        rank = (ranks * tp).max(dim=-1).values

        # mrr will contain 'inf' values if target is not in top k scores -> setting it to 0
        mrr = 1 / rank
        mrr[mrr == float('inf')] = 0
        return mrr

    def name(self):
        return f"MRR"
