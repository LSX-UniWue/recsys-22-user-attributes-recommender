from typing import Any, Optional, Tuple, Union, Dict, List

import pytorch_lightning as pl
import torch


# TODO (AD): For Transformer models we could evaluate the whole sequence at once


class RecallAtMetric(pl.metrics.Metric):
    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):

        super(RecallAtMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k
        self.add_state("recall", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")


    def update(self,
               prediction: torch.Tensor,
               target: torch.Tensor
               ) -> torch.Tensor:
        """
        :param prediction: the scores of all items, the first one is the positive item, all others are negatives
        :param target: the target label tensor
        :return: the recall at specified k
        """
        sorted_indices = torch.topk(prediction, k=self._k)[1]
        target_expanded = target.view(-1, 1).expand(-1, self._k)

        # FIXME (AD) make it work for target vectors with size > 1
        tp = sorted_indices.eq(target_expanded).sum(dim=-1).to(dtype=torch.float)
        fn = torch.ones(tp.size(), device=tp.device) - tp

        recall = tp / (tp + fn)
        self.recall += recall.sum()
        self.count += recall.size()[0]

    def compute(self):
        return self.recall / self.count


class MRRAtMetric(pl.metrics.Metric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False):

        super(MRRAtMetric, self).__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k

        self.add_state("mrr", torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")


    def update(self,
               prediction: torch.Tensor,
               target: torch.Tensor
               ) -> torch.Tensor:
        """
        :param prediction: the scores of all items, the first one is the positive item, all others are negatives
        :param target: the target label tensor
        :return: the recall at specified k
        """
        sorted_indices = torch.topk(prediction, k=self._k)[1]
        target = target.view(-1, 1).expand(-1, self._k)

        rank = torch.topk((sorted_indices.eq(target) * torch.arange(1, self._k + 1, dtype=torch.float, device=target.device)), k=1)[0]

        # mrr will contain 'inf' values if target is not in topk scores -> setting to 0
        mrr = 1 / rank
        mrr[mrr == float('inf')] = 0

        self.mrr += mrr.sum()
        self.count += mrr.size()[0]

    def compute(self):
        return self.mrr / self.count


if __name__ == "__main__":
    prediction = torch.tensor([
        [0.5, 0.12, 0.3, 0.18],
        [0.5, 0.12, 0.3, 0.18],
    ])
    target = torch.tensor([[2], [0]])

    metric = MRRAtMetric(k=3)
    metric(prediction, target)

    target = torch.tensor([[1], [1]])
    metric(prediction, target)
    print(metric.compute())
