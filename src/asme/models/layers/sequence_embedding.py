from functools import partial

import torch
from torch import nn


def _max_pooling(tensor: torch.Tensor
                 ) -> torch.Tensor:
    return tensor.max(dim=-2)[0]


def _identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


class PooledSequenceElementsLayer(nn.Module):
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


class SequenceElementsEmbeddingLayer(nn.Module):
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
            self.pooling = PooledSequenceElementsLayer(embedding_pooling_type)
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