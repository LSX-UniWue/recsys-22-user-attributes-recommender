from typing import Any, Optional

import pytorch_lightning as pl
import torch


class RecallAtMetric(pl.metrics.TensorMetric):

    def __init__(
            self,
            reduction: str = 'elementwise_mean',
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        super().__init__(name='recall_at',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)
        self.reduction = reduction

    def forward(self,
                scores: torch.Tensor,
                ) -> torch.Tensor:
        """
        :param scores: the scores of all items, the first one is the positive item, all others are negatives
        :return: the recall at specified k
        """
        test = torch.topk(scores, k=5)
        # TODO

        pass



if __name__ == '__main__':
    test = torch.randn([101, 2, 32])
    test = torch.softmax(test, dim=2)

    print(test[-1])

    recall_metric = RecallAtMetric()

    recall = recall_metric.forward(test)
    print(recall)
