from typing import Tuple

import pytest
import torch
from util import build_sample, EPSILON

from metrics.ranking.ndcg import NormalizedDiscountedCumulativeGain


def get_single_item_recommendation_samples():
    return [
        build_sample(torch.tensor([[5, 4, 3, 2, 1]]), torch.tensor([0]), None, 1, torch.tensor(1.)),
        build_sample(torch.tensor([[4, 5, 3, 2, 1]]), torch.tensor([0]), None, 1, torch.tensor(0.)),
        build_sample(torch.tensor([[5, 4, 3, 2, 1]]), torch.tensor([0]), None, 3, torch.tensor(1.)),
        build_sample(torch.tensor([[4, 5, 3, 2, 1]]), torch.tensor([0]), None, 3, torch.tensor(0.6309)),
        build_sample(torch.tensor([[3, 4, 5, 2, 1]]), torch.tensor([0]), None, 3, torch.tensor(0.5)),
        build_sample(torch.tensor([[5, 4, 3, 2, 1]]), torch.tensor([0]), None, 5, torch.tensor(1.)),
        build_sample(torch.tensor([[4, 5, 3, 2, 1]]), torch.tensor([0]), None, 5, torch.tensor(0.6309)),
        build_sample(torch.tensor([[3, 4, 5, 2, 1]]), torch.tensor([0]), None, 5, torch.tensor(0.5)),
        build_sample(torch.tensor([[2, 3, 4, 5, 1]]), torch.tensor([0]), None, 5, torch.tensor(0.4306)),
        build_sample(torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([0]), None, 5, torch.tensor(0.3868)),
    ]


def get_multiple_item_recommendation_samples():
    return [
        build_sample(torch.tensor([[0, 0, 0, 4, 5]]), torch.tensor([[3, 4]]), None, 2, torch.tensor(1.)),
        build_sample(torch.tensor([[0, 0, 4, 5, 0]]), torch.tensor([[3, 4]]), None, 2,
                     torch.tensor(1. / (1. + 0.6309))),
        build_sample(torch.tensor([[0, 4, 5, 0, 0]]), torch.tensor([[3, 4]]), None, 2, torch.tensor(0.)),
        build_sample(torch.tensor([[0, 0, 0, 4, 5]]), torch.tensor([[3, 4]]), torch.tensor([[0, 1]]), 2,
                     torch.tensor(1.)),
        build_sample(torch.tensor([[0, 0, 0, 4, 5]]), torch.tensor([[3, 4]]), torch.tensor([[1, 0]]), 2,
                     torch.tensor(0.6309)),
        build_sample(torch.tensor([[0, 3, 4, 0, 5]]), torch.tensor([[2, 3, 4]]), None, 2, torch.tensor(1.)),
        build_sample(torch.tensor([[0, 3, 4, 0, 5]]), torch.tensor([[2, 3, 4]]), None, 3,
                     torch.tensor((1. + 0.6309) / (1. + 0.6309 + 0.5))),
        build_sample(torch.tensor([[0, 3, 4, 0, 5]]), torch.tensor([[2, 3, 4]]), torch.tensor([[0, 1, 0]]), 2,
                     torch.tensor(0.)),
        build_sample(torch.tensor([[0, 3, 4, 0, 5]]), torch.tensor([[2, 3, 4]]), torch.tensor([[0, 1, 1]]), 2,
                     torch.tensor(1. / (1. + 0.6309))),
    ]


class TestNDCG:

    @pytest.mark.parametrize("sample", get_single_item_recommendation_samples())
    def test_single_item_recommendation(self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]):
        predictions, target, mask, k, value = sample
        instance = NormalizedDiscountedCumulativeGain(k=k)
        instance.update(predictions, target, mask)

        assert torch.sum(torch.abs(instance.compute() - value)) < EPSILON

    @pytest.mark.parametrize("sample", get_multiple_item_recommendation_samples())
    def test_multiple_item_recommendation(self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]):
        predictions, target, mask, k, value = sample
        instance = NormalizedDiscountedCumulativeGain(k=k)
        instance.update(predictions, target, mask)

        assert torch.sum(torch.abs(instance.compute() - value)) < EPSILON
