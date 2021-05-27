from typing import Optional, Dict
from functools import partial

import torch
import torch.nn as nn

from asme.models.layers.layers import SequenceElementsRepresentationLayer


def _max_pooling(tensor: torch.Tensor
                 ) -> torch.Tensor:
    return tensor.max(dim=-2)[0]


def _identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


class SequenceElementsEmbeddingLayer(SequenceElementsRepresentationLayer):
    """
    Computes an embedding for every element in the sequence.
    """
    def __init__(self,
                 item_voc_size: int,
                 embedding_size: int):
        super().__init__()

        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(num_embeddings=item_voc_size, embedding_dim=embedding_size)

    def forward(self,
                sequence: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                **kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:

        embedded_sequence = self.embedding(sequence)

        return embedded_sequence


class PooledSequenceElementsRepresentation(nn.Module):
    """
    A stateless module that provides pooling functionality over embedded sequences.
    """
    def __init__(self, pooling_type: str):
        """

        :param pooling_type: the type of pooling to perform, can be either: `max`, `sum`, or `mean`.
        """
        super().__init__()

        self.pooling_function = {
            'max': _max_pooling,
            'sum': partial(torch.sum, dim=-2),
            'mean': partial(torch.mean, dim=-2)
        }[pooling_type]

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Applies the pooling function to the sequence.

        :param sequence: a sequence with embedded elements. :math:`(N, S, H)`
        :return: the sequence representation after the pooling :math:`(N, H)`
        """
        return self.pooling_function(sequence)
