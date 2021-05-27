import math
import abc
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Tuple
from abc import ABC
from functools import partial

import torch
from torch import nn

from asme.models.layers.sequence_embedding import PooledSequenceElementsRepresentation


class SequenceElementsRepresentationLayer(ABC, nn.Module):
    """
    Base class for modules that embed the elements of a sequence.
    """
    @abc.abstractmethod
    def forward(self,
                sequence: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                **kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        :param sequence: a sequence tensor with item ids. :math:`(N, S)` or :math:`(N, S, BS)`
        :param padding_mask: a mask that contains positions of padding tokens. :math:`(N, S)`
        :param kwargs: attributes that can be used to contextualize the sequence

        :return: a sequence with embedded elements. :math:`(N, S, H)`
        """
        pass


class SequenceRepresentationLayer(ABC, nn.Module):
    @abc.abstractmethod
    def forward(self,
                sequence: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                **kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param sequence: an embedded sequence tensor. :math:`(N, S, H)`
        :param padding_mask: a mask that contains positions of padding tokens. :math:`(N, S)`
        :param kwargs: attributes that can be used to contextualize the sequence representation.

        :return: a sequence representation. :math:`(N, S, R)`
        """
        pass


class SequenceRepresentationModifierLayer(ABC, nn.Module):
    @abc.abstractmethod
    def forward(self,
                sequence: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                **kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param sequence: a sequence tensor. :math:`(N, S, R)`
        :param padding_mask: a mask that contains positions of padding tokens. :math:`(N, S)`
        :param kwargs: attributes that can be used to contextualize the sequence

        :return: a sequence with embedded elements. :math:`(N, S, T)`
        """
        pass


class ProjectionLayer(ABC, nn.Module):
    @abc.abstractmethod
    def forward(self,
                sequence_representation: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                positive_samples: Optional[torch.Tensor] = None,
                negative_samples: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param sequence_representation: a sequence tensor. :math:`(N, S, T)`
        :param padding_mask: a mask that contains positions of padding tokens. :math:`(N, S)`
        :param positive_samples: a tensor with positive sample item ids to score. :math:`(N, P)`
        :param negative_samples: a tensor with negative sample item ids o score. :math:`(N, NI)`

        :return: a score for each (provided) item.
        if no positive_samples and negative samples are provided a tensor of :math:`(N, I)` is returned
        if positive samples are provided :math:`(N, PI)` or
        if positive and negative samples are provided a tuple of two tensors of shape :math:`(N, PI)`, :math:`(N, NI)`
        """
        pass


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

        if embedding_pooling_type:
            self.pooling = PooledSequenceElementsRepresentation(embedding_pooling_type)
        else:
            self.pooling = None

        self.embedding_mode = embedding_pooling_type
        self.embedding = nn.Embedding(num_embeddings=item_voc_size,
                                      embedding_dim=self.embedding_size)

    def get_weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self,
                items: torch.Tensor,
                flatten: bool = True
                ) -> torch.Tensor:
        embedding = self.embedding(items)

        # this is a quick hack, if a module needs the embeddings of a single item
        if self.pooling:
            embedding = self.pooling(embedding)
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
