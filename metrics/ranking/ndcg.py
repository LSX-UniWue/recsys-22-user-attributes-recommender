import torch

from metrics.ranking.common import calc_dcg, reverse_cumsum
from metrics.ranking.ranking_metric import RankingMetric


class NormalizedDiscountedCumulativeGain(RankingMetric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super(NormalizedDiscountedCumulativeGain, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("ndcg", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('count', torch.tensor(0.), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor = None
                ) -> None:
        """
        :param prediction: the scores of all items, the first one is the positive item, all others are negatives :math `(N, I)`
        :param target: the target label tensor :math `(N, T)`
        """
        dcg = calc_dcg(prediction, target, self._k, mask)

        # Number of items that are expected to be predicted by the model
        target_size = target.size()
        batch_size = target_size[0]

        # Calculate the number of relevant items per sample. If a mask is provided, we only consider the unmasked items
        if mask is not None:
            number_of_relevant_items = mask.sum(dim=1)
        else:
            # If there is only a single dimension, we are in single-item prediction mode
            if len(target_size) == 1:
                number_of_relevant_items = torch.ones((batch_size,))
            # If there is more than one dimension but no mask, all items in the target tensor are considered
            else:
                number_of_relevant_items = torch.full((batch_size,), target_size[1])

        # The model can only suggest k items, therefore the ideal DCG should be calculated using k as an upper bound
        # on the number of relevant items to ensure comparability
        number_of_relevant_items = torch.clamp(number_of_relevant_items, max=self._k)

        # FIXME: The following code seems unnecessarily complicated, is there a nicer and more concise way to achieve the same result?
        number_of_relevant_items = torch.unsqueeze(number_of_relevant_items, dim=1)
        max_number_of_relevant_items = max(number_of_relevant_items.long()).item()
        # This lets us use relevant_positions as an index in the following
        relevant_positions = torch.cat([torch.arange(0, batch_size).unsqueeze(dim=1) ,number_of_relevant_items - 1],
                                       dim=1).long()

        positions_values = torch.zeros((batch_size, max_number_of_relevant_items))
        # This just places 1s at the locations indicated by relevant_positions
        positions_values[relevant_positions[:, 0], relevant_positions[:, 1]] = 1
        # At this point, position_values is filled with as may ones per sample as there are relevant items.
        # The remaining elements per sample are 0s.
        positions_values = reverse_cumsum(positions_values, dim=1)

        # Calculate the ideal DCG with the values calculated above.
        idcg = torch.sum(1 / torch.log2(positions_values * torch.arange(2, max_number_of_relevant_items + 2)), dim=1)

        # Update cumulative ndcg and measurements count
        self.ndcg += (dcg / idcg).sum()
        self.count += dcg.size()[0]

    def compute(self):
        return self.ndcg / self.count

    def name(self):
        return f"ndcg_at_{self._k}"

