from typing import Union

import torch


def build_sample(predictions: torch.Tensor,
                 target: torch.Tensor,
                 mask: Union[torch.Tensor, None],
                 k: int,
                 value: torch.Tensor):
    return predictions.float(), target, mask, k, value


EPSILON = 10e-4
