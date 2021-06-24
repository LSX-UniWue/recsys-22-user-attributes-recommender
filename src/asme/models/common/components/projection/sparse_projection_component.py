from typing import Union, Tuple

import torch
from asme.models.common.layers.data.sequence import ModifiedSequenceRepresentation
from asme.models.common.layers.layers import ProjectionLayer
from torch import nn


class SparseProjectionComponent(ProjectionLayer):

    """
    a projection layer that uses embeddings to calculate the projection of a subset of the item set
    """

    def __init__(self,
                 item_vocab_size: int,
                 embedding_size: int,
                 ):
        super().__init__()

        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        self.W2 = nn.Embedding(item_vocab_size, embedding_size)
        self.b2 = nn.Embedding(item_vocab_size, 1)

        # init weights
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        #if not self.training:
        #    w2 = pos_w2.squeeze()
        #    b2 = pos_b2.squeeze()
        #    w2 = w2.permute(1, 0, 2)
        #    return torch.matmul(x, w2).sum(dim=1) + b2
        input_sequence = modified_sequence_representation.input_sequence
        positive_samples = input_sequence.get_attribute("positive_samples")
        negative_samples = input_sequence.get_attribute("negative_samples")

        sequence_representation = modified_sequence_representation.modified_encoded_sequence
        sequence_representation = sequence_representation.unsqueeze(2)
        res_pos = self._calc_scores(positive_samples, sequence_representation)
        if negative_samples is None:
            return res_pos

        # negative items
        res_negative = self._calc_scores(negative_samples, sequence_representation)
        return res_pos, res_negative

    def _calc_scores(self,
                     item: torch.Tensor,
                     sequence_representation: torch.Tensor
                     ) -> torch.Tensor:
        w2 = self.W2(item)  # (N, I, F_D)
        b2 = self.b2(item)  # (N, I, 1)

        # TODO: use torch.einsum('nif,nf -> ni', w2, x) + b2.squeeze()
        return torch.baddbmm(b2, w2, sequence_representation).squeeze()