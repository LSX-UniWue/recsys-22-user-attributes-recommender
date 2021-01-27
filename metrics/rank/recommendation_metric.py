from abc import abstractmethod

import pytorch_lightning as pl
import torch


class RecommendationMetric(pl.metrics.Metric):

    def update(self,
               predictions: torch.Tensor,
               target: torch.Tensor,
               mask: torch.Tensor
               ) -> None:
        self._update(predictions, target, mask)

    @abstractmethod
    def _update(self,
                predictions: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor
                ) -> None:
        pass