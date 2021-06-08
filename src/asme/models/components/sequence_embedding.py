from typing import Optional, Dict

import torch
from torch import nn

from asme.models.layers.layers import SequenceElementsRepresentationLayer


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