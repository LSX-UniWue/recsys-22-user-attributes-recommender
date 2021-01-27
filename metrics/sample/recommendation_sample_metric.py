from abc import abstractmethod

import pytorch_lightning as pl
import torch


class RecommendationSampleMetric(pl.metrics.Metric):

    def update(self,
               predictions: torch.Tensor,
               positive_item_mask: torch.Tensor
               ) -> None:
        self._update(predictions, positive_item_mask)

    @abstractmethod
    def _update(self,
                predictions: torch.Tensor,
                positive_item_mask: torch.Tensor
                ) -> None:
        pass

    def name(self):
        """
        Returns the name that identifies this metric.

        :return: the name.
        """
        pass