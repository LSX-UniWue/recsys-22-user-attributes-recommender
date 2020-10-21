from abc import ABC

import torch
from torch import nn as nn


ACTIVATION_FUNCTION_KEY_RELU = 'relu'


def get_activation_layer(activation_fn_name: str) -> nn.Module:
    """
    :param activation_fn_name: the name of the activation function
    :return: the torch layer for the specified layer name
    """
    return {
        'identity': nn.Identity(),
        ACTIVATION_FUNCTION_KEY_RELU: nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid()
    }[activation_fn_name]


class MatrixFactorizationLayer(nn.Linear, ABC):
    """
    a matrix factorization layer
    """

    def __init__(self,
                 _weight: torch.Tensor,
                 bias: bool = False
                 ):
        super().__init__(_weight.size(0), _weight.size(1), bias=bias)
        self.weight = _weight
