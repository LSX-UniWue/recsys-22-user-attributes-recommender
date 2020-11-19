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
        'sigmoid': nn.Sigmoid(),
        'gelu': nn.GELU(),
        'glu': nn.GLU()
    }[activation_fn_name]


def truncated_normal(tensor: torch.Tensor,
                     mean: float = 0.0,
                     stddev: float = 1.0
                     ) -> torch.Tensor:
    """" the truncated normal distribution """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(stddev).add_(mean)
    return tensor
