import torch
from torch import nn as nn


class MatrixFactorizationLayer(nn.Linear):
    """
    a matrix factorization layer
    """

    def __init__(self,
                 _weight: torch.Tensor):
        super().__init__(_weight.size(0), _weight.size(1), bias=False)
        self.weight = _weight
