from abc import ABC
from functools import partial

import torch
from torch import nn as nn


def _max_pooling(tensor: torch.Tensor
                 ) -> torch.Tensor:
    return tensor.max(dim=-2)[0]


class ItemEmbedding(nn.Module):
    """
    embedding to use for the items
    handles multiple items per sequence step, by averaging, summing or max the single embeddings
    """

    def __init__(self,
                 item_voc_size: int,
                 embedding_size: int,
                 embedding_pooling_type: str = None
                 ):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding_mode = embedding_pooling_type
        self.embedding = nn.Embedding(num_embeddings=item_voc_size,
                                      embedding_dim=self.embedding_size)

        self.embedding_flatten = {
            None: lambda x: x,
            'max': partial(_max_pooling),
            'sum': partial(torch.sum, dim=-2),
            'mean': partial(torch.mean, dim=-2)
        }[self.embedding_mode]

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
