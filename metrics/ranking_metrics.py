from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch


class RecallAtMetric(pl.metrics.metric.TensorCollectionMetric):
    def __init__(
            self,
            k: int,
            reduction: str = 'elementwise_mean',
            reduce_group: Any = None,
            reduce_op: Any = None
    ):
        super().__init__(name='recall_at',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.reduction = reduction
        self._k = k

    def forward(self,
                scores: torch.Tensor,
                target: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param scores: the scores of all items, the first one is the positive item, all others are negatives
        :param target: the target label tensor
        :return: the recall at specified k
        """
        sorted_indices = torch.topk(scores, k=self._k)[1]
        target = target.view(-1, 1).expand(-1, self._k)

        tp = sorted_indices.eq(target).sum().to(dtype=torch.float)
        tpfn = torch.as_tensor(target.size()[0], dtype=torch.float)

        return tp, tpfn, tp / tpfn
