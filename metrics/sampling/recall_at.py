import torch

from metrics.sampling.sampling_metric import SamplingMetric


class RecallAtNegativeSamples(SamplingMetric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("recall", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state('count', torch.tensor(0.), dist_reduce_fx="sum")

    def _update(self,
                prediction: torch.Tensor,
                positive_item_mask: torch.Tensor
                ) -> None:
        """

        :param prediction: the logits for I items :math`(N, I)`
        :param positive_item_mask: a mask where a 1 indices that the item at this index is relevant :math`(N, I)`
        :return:
        """
        # get the indices of the top k predictions
        predicted_id = torch.argsort(prediction, descending=True)
        # limit the prediction to the top k
        predicted_id = predicted_id[:, :self._k]

        # select the mask for each of the indices, this is than the tp
        tp = positive_item_mask.gather(1, predicted_id)
        tp = tp.sum(1)  # get the total number of positive items

        all_relevant_items = positive_item_mask.sum(1)
        recall = tp / all_relevant_items
        self.recall += recall.sum()
        self.count += prediction.size()[0]

    def compute(self):
        return self.recall / self.count

    def name(self):
        return f"recall_at_{self._k}/sampled"
