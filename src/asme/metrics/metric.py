from abc import abstractmethod

import pytorch_lightning as pl
import torch


class RankingMetric(pl.metrics.Metric):

    """
    base class to implement metrics in the framework
    """

    def update(self,
               predictions: torch.Tensor,
               positive_item_mask: torch.Tensor,
               metric_mask: torch.Tensor = None
               ) -> None:
        """
        :param predictions: the prediction logits :math `(N, I)`
        :param positive_item_mask: the positive item mask 1 if the item at index i is relevant :math `(N)`
        :param metric_mask: a mask to mask single item in the prediction :math `(N)`

        where N is the batch size and I the item size to evaluate (can be the item vocab size)
        """
        if metric_mask is None:
            metric_mask = torch.ones_like(positive_item_mask)
        # just call the correct update function
        self._update(predictions, positive_item_mask, metric_mask)

    @abstractmethod
    def _update(self,
                predictions: torch.Tensor,
                positive_item_mask: torch.Tensor,
                metric_mask: torch.Tensor
                ) -> None:

        pass

    @abstractmethod
    def name(self):
        """
        Returns the name that identifies this metric.

        :return: the name.
        """
        pass
