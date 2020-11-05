import pytorch_lightning as pl
import torch


def get_tp(predictions: torch.Tensor,
           target: torch.Tensor,
           mask: torch.Tensor,
           k: int
           ) -> torch.Tensor:
    sorted_indices = torch.topk(predictions, k=k).indices
    sorted_indices = torch.repeat_interleave(sorted_indices, mask.size()[-1], dim=0)

    target_expanded = target.view(-1, 1).expand(-1, k)
    tp_mask = torch.repeat_interleave(mask.view(-1, 1), k, dim=1)

    tp = (sorted_indices.eq(target_expanded) * tp_mask).sum(dim=-1).to(dtype=torch.float)
    return tp.view(mask.size()).sum(-1)


def calc_precision(prediction: torch.Tensor,
                   target: torch.Tensor,
                   k: int,
                   mask: torch.Tensor = None
                   ) -> torch.Tensor:
    if len(target.size()) == 1:
        # single dimension, unsqueeze it
        target = torch.unsqueeze(target, dim=1)

    if mask is None:
        mask = torch.ones(target.size())

    tp = get_tp(predictions=prediction, target=target, mask=mask, k=k)

    return tp / k


def calc_recall(prediction: torch.Tensor,
                target: torch.Tensor,
                k: int,
                mask: torch.Tensor = None
                ) -> torch.Tensor:
    if len(target.size()) == 1:
        # single dimension, unsqueeze it
        target = torch.unsqueeze(target, dim=1)

    if mask is None:
        mask = torch.ones(target.size(), device=prediction.device)

    tp = get_tp(predictions=prediction, target=target, mask=mask, k=k)
    fn = mask.sum(-1) - tp

    return tp / (tp + fn)


class F1AtMetric(pl.metrics.Metric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False
                 ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("f1", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")
        self._k = k

    def update(self,
               prediction: torch.Tensor,
               target: torch.Tensor,
               mask: torch.Tensor = None,
               ):
        recall = calc_recall(prediction, target, self._k, mask=mask)
        precision = calc_precision(prediction, target, self._k, mask=mask)

        f1 = 2 * recall * precision / (recall + precision)
        f1[torch.isnan(f1)] = 0.0

        self.f1 += f1.sum()
        self.count += f1.size()[0]

    def compute(self):
        return self.f1 / self.count


class PrecisionAtMetric(pl.metrics.Metric):

    def __init__(self,
                 k: int,
                 dist_sync_on_step: bool = False
                 ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self._k = k
        self.add_state("precision", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0), dist_reduce_fx="sum")

    def update(self,
               prediction: torch.Tensor,
               target: torch.Tensor,
               mask: torch.Tensor = None,
               ):
        """
        :param prediction: the scores of all items :math `(N, I)`
        :param target: the target label tensor :math `(N, T)` or :math `(N)`
        :param mask: the mask to apply, iff no mask is provided all targets are used for calculating the metric
        :math `(N, I)`
        """
        precision = calc_precision(prediction, target, self._k, mask=mask)
        self.precision += precision.sum()
        self.count += precision.size()[0]

    def compute(self):
        return self.precision / self.count


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
               target: torch.Tensor,
               mask: torch.Tensor = None,
               ):
        """
        :param prediction: the scores of all items :math `(N, I)`
        :param target: the target label tensor :math `(N, T)` or :math `(N)`
        :param mask: the mask to apply, iff no mask is provided all targets are used for calculating the metric
        :math `(N, I)`
        """

        recall = calc_recall(prediction, target, self._k, mask=mask)
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
        :param prediction: the scores of all items, the first one is the positive item, all others are negatives :math `(N, I)`
        :param target: the target label tensor :math `(N, T)`
        :return: the mrr at specified k
        """
        sorted_indices = torch.topk(prediction, k=self._k).indices
        target = target.view(-1, 1).expand(-1, self._k)

        rank = torch.topk((sorted_indices.eq(target) * torch.arange(1, self._k + 1, dtype=torch.float, device=target.device)), k=1).values

        # mrr will contain 'inf' values if target is not in topk scores -> setting to 0
        mrr = 1 / rank
        mrr[mrr == float('inf')] = 0

        self.mrr += mrr.sum()
        self.count += mrr.size()[0]

    def compute(self):
        return self.mrr / self.count


if __name__ == "__main__":
    test_prediction = torch.tensor([
        [0.5, 0.12, 0.3, 0.18],  # ranking: 0, 2, 3, 1
        [0.5, 0.12, 0.6, 0.18],  # ranking: 2, 0, 3, 1
    ])
    test_target = torch.tensor([[0], [12]])  # assume arbitrary masking id
    #mask = torch.tensor([[1, 1], [1, 1]])
    # k=3
    # max_target_size = 2
    #
    # if max_target_size > k:
    #     raise Exception()
    #
    # sorted_indices = torch.topk(prediction, k=k)[1]
    # sorted_indices = torch.repeat_interleave(sorted_indices, max_target_size, dim=0)
    #
    #
    # print(sorted_indices)
    #
    # target = torch.tensor([[2, 1], [0, 3]])  # assume arbitrary masking id
    # mask = torch.tensor([[1, 1], [1, 0]])  # mask marks last entry in target as padded
    #
    #
    # target = target.view(-1, 1)
    # print(target)
    # target = target.expand(-1, k)
    # print(target)
    #
    # rank = torch.topk((sorted_indices.eq(target) * torch.arange(1, k + 1, dtype=torch.float, device=target.device)), k=k)[0]
    # print(rank)
    #
    # rank = rank * mask.view([1, -1])
    # print(rank)

    metric = RecallAtMetric(k=1)
    metric(test_prediction, test_target)
    print(metric.compute())

