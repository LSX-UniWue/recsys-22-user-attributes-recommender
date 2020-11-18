from abc import ABC
from functools import partial
from typing import Callable

import torch
from torch import nn as nn


class ItemEmbedding(nn.Module):
    """
    embedding to use for the items
    handles multiple items per sequence step, by averaging or summing the single embeddings
    """

    def __init__(self,
                 item_voc_size: int,
                 embedding_size: int,
                 embedding_mode: str = None,
                 init_weights_fnc: Callable[[torch.Tensor], None] = None
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding_mode = embedding_mode
        self.embedding = nn.Embedding(num_embeddings=item_voc_size,
                                      embedding_dim=self.embedding_size)

        self.embedding_flatten = {
            None: lambda x: x,
            'sum': partial(torch.sum, dim=-2),
            'mean': partial(torch.mean, dim=-2)
        }[self.embedding_mode]

        self.init_weights(init_weights_fnc)

    def init_weights(self, init_weights_fnc: Callable[[torch.Tensor], None]):
        if init_weights_fnc is None:
            initrange = 0.1
            self.embedding.weight.data.uniform_(- initrange, initrange)
        else:
            init_weights_fnc(self.embedding.weight)

    def forward(self,
                items: torch.Tensor,
                flatten: bool = True
                ) -> torch.Tensor:
        embedding = self.embedding(items)

        # this is a quick hack, if a module needs the embeddings of a single item
        if flatten:
            embedding = self.embedding_flatten(embedding)
        return embedding


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
