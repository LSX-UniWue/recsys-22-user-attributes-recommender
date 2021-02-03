from abc import abstractmethod

import pytorch_lightning as pl
import torch


class RankingMetric(pl.metrics.Metric):

    def update(self,
               predictions: torch.Tensor,
               targets: torch.Tensor,
               mask: torch.Tensor
               ) -> None:
        self._update(predictions, targets, mask)

    @abstractmethod
    def _update(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: torch.Tensor
                ) -> None:
        pass

    @abstractmethod
    def name(self):
        """
        Returns the name that identifies this metric.

        :return: the name.
        """
        pass
