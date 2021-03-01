from typing import Union

import torch


def build_sample(predictions: torch.Tensor,
                 target: torch.Tensor,
                 k: int,
                 value: torch.Tensor):
    return predictions.float(), target, k, value


EPSILON = 10e-4
