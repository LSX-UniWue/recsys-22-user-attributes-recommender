import torch

from metrics.ranking.ranking_metric import RankingMetric


class MRRAtMetric(RankingMetric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super(MRRAtMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("mrr", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor
                ) -> None:
        """
        :param prediction: the scores of all items, the first one is the positive item, all others are negatives :math `(N, I)`
        :param target: the target label tensor :math `(N, T)`
        :return: the mrr at specified k
        """
        sorted_indices = torch.topk(prediction, k=self._k).indices
        target = target.view(-1, 1).expand(-1, self._k)

        rank = torch.topk(
            (sorted_indices.eq(target) * torch.arange(1, self._k + 1, dtype=torch.float, device=target.device)),
            k=1).values

        # mrr will contain 'inf' values if target is not in top k scores -> setting it to 0
        mrr = 1 / rank
        mrr[mrr == float('inf')] = 0

        self.mrr += mrr.sum()
        self.count += mrr.size()[0]

    def compute(self):
        return self.mrr / self.count

    def name(self):
        return f"mrr_at_{self._k}"
