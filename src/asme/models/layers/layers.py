import math
from abc import ABC
from functools import partial
from typing import Optional, Tuple, Union

import torch
from torch import nn

from asme.models.sequence_recommendation_model import ProjectionLayer


def _max_pooling(tensor: torch.Tensor
                 ) -> torch.Tensor:
    return tensor.max(dim=-2)[0]


def _identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


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
            None: _identity,
            'max': _max_pooling,
            'sum': partial(torch.sum, dim=-2),
            'mean': partial(torch.mean, dim=-2)
        }[self.embedding_mode]

    def get_weight(self) -> torch.Tensor:
        return self.embedding.weight

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


PROJECT_TYPE_LINEAR = 'linear'


class LinearProjectionLayer(ProjectionLayer):

    """
    A projection layer that uses a linear layer to project the sequence representation into a score for each item
    """

    def __init__(self,
                 hidden_size: int,
                 item_vocab_size: int):
        super().__init__()

        self.linear = nn.Linear(hidden_size, item_vocab_size)

    def forward(self,
                sequence_representation: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                positive_samples: Optional[torch.Tensor] = None,
                negative_samples: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.linear(sequence_representation)


class ItemEmbeddingProjectionLayer(ProjectionLayer):

    """
    A projection layer that uses the item embedding matrix to transform the sequence representation into the item space

    TODO: currently the sequence representation size must be the same as the item space size
    """

    def __init__(self,
                 item_vocab_size: int,
                 embedding: nn.Embedding
                 ):
        super().__init__()

        self.item_vocab_size = item_vocab_size

        self.embedding = embedding
        self.output_bias = nn.Parameter(torch.Tensor(self.item_vocab_size))

        self.init_weights()

    def init_weights(self):
        bound = 1 / math.sqrt(self.item_vocab_size)
        nn.init.uniform_(self.output_bias, -bound, bound)

    def forward(self,
                sequence_representation: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                positive_samples: Optional[torch.Tensor] = None,
                negative_samples: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense = torch.matmul(sequence_representation, self.embedding.weight.transpose(0, 1))  # (S, N, I)
        return dense + self.output_bias


def build_projection_layer(project_type: str,
                           transformer_hidden_size: int,
                           item_voc_size: int,
                           embedding: nn.Embedding
                           ) -> ProjectionLayer:
    if project_type == PROJECT_TYPE_LINEAR:
        return LinearProjectionLayer(transformer_hidden_size, item_voc_size)

    if project_type == 'transpose_embedding':
        return ItemEmbeddingProjectionLayer(item_voc_size, embedding)

    raise KeyError(f'{project_type} invalid projection layer')
