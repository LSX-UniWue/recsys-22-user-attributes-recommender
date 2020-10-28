from abc import ABC

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
        if embedding_mode is not None:
            embedding = nn.EmbeddingBag(num_embeddings=item_voc_size,
                                        embedding_dim=self.embedding_size,
                                        mode=embedding_mode)
        else:
            embedding = nn.Embedding(num_embeddings=item_voc_size,
                                     embedding_dim=self.embedding_size)
        self.embedding = embedding
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(- initrange, initrange)

    def forward(self,
                items: torch.Tensor
                ) -> torch.Tensor:
        return self.embedding(items)


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