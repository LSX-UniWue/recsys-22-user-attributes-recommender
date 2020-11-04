from abc import ABC
from functools import partial

import torch
from torch import nn as nn


class ItemEmbedding(nn.Module):

    def __init__(self,
                 item_voc_size: int,
                 embedding_size: int,
                 embedding_mode: str = None,
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
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(- initrange, initrange)

    def forward(self,
                items: torch.Tensor
                ) -> torch.Tensor:
        embedding = self.embedding(items)
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