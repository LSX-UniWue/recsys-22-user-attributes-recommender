from typing import Any, Optional, Tuple, Union, Dict, List

import pytorch_lightning as pl
import torch
from pytorch_lightning import EvalResult

from metrics.base import AggregatingMetricTrait


class RecallAtMetric(pl.metrics.metric.TensorMetric, AggregatingMetricTrait):
    def __init__(self,
                 k: int,
                 reduction: str = 'elementwise_mean',
                 reduce_group: Any = None,
                 reduce_op: Any = None):

        super().__init__(name='recall_at',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.reduction = reduction
        self._k = k

    def forward(self,
                scores: torch.Tensor,
                target: torch.Tensor
                ) -> torch.Tensor:
        """
        :param scores: the scores of all items, the first one is the positive item, all others are negatives
        :param target: the target label tensor
        :return: the recall at specified k
        """
        sorted_indices = torch.topk(scores, k=self._k)[1]
        target_expanded = target.view(-1, 1).expand(-1, self._k)

        # FIXME (AD) make it work for target vectors with size > 1
        tp = sorted_indices.eq(target_expanded).sum(dim=-1).to(dtype=torch.float)
        fn = torch.ones(tp.size()) - tp

        tpfn = tp + fn

        return tp / tpfn

    def on_step_end(self, prediction: torch.Tensor, target: torch.Tensor, result: EvalResult) -> None:
        recall_at_k = self(prediction, target)

        result.log(f"recall_at_{self._k}", recall_at_k, prog_bar=True)

    def on_epoch_end(self, outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
                     result: EvalResult) -> None:
        result.log(f"recall_at_{self._k}", outputs[f"recall_at_{self._k}"].mean(), prog_bar=True)


if __name__ == "__main__":
    prediction = torch.tensor([
        [0.5, 0.12, 0.3, 0.18],
        [0.5, 0.12, 0.3, 0.18],
    ])
    target = torch.tensor([[0], [1]])

    metric = RecallAtMetric(k=3)
    result = metric(prediction, target)

    print(result)


class MRRAtMetric(pl.metrics.metric.TensorMetric, AggregatingMetricTrait):

    def __init__(self,
                 k: int,
                 reduction: str = 'elementwise_mean',
                 reduce_group: Any = None,
                 reduce_op: Any = None):

        super().__init__(name='mrr_at',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.reduction = reduction
        self._k = k

    def forward(self,
                scores: torch.Tensor,
                target: torch.Tensor
                ) -> torch.Tensor:
        """
        :param scores: the scores of all items, the first one is the positive item, all others are negatives
        :param target: the target label tensor
        :return: the recall at specified k
        """
        sorted_indices = torch.topk(scores, k=self._k)[1]
        target = target.view(-1, 1).expand(-1, self._k)

        rank = torch.topk((sorted_indices.eq(target) * torch.arange(1, self._k + 1, dtype=torch.float)), k=1)[0]

        # mrr will contain 'inf' values if target is not in topk scores -> setting to 0
        mrr = 1 / rank
        mrr[mrr == float('inf')] = 0

        return mrr

    def on_step_end(self, prediction: torch.Tensor, target: torch.Tensor, result: EvalResult) -> None:
        mrr = self(prediction, target)
        result.log(f"mrr_at_{self._k}", mrr, prog_bar=True)

    def on_epoch_end(self, outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
                     result: EvalResult) -> None:
        result.log(f"mrr_at_{self._k}", outputs[f"mrr_at_{self._k}"].mean(), prog_bar=True)
