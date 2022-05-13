import torch
from asme.core.models.common.layers.data.sequence import SequenceRepresentation, ModifiedSequenceRepresentation
from asme.core.models.common.layers.layers import SequenceRepresentationModifierLayer
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.data.datasets import USER_ENTRY_NAME
from torch import nn


class UserEmbeddingConcatModifier(SequenceRepresentationModifierLayer):

    """
    a sequence representation modifier that cats a user embedding to the sequence representation
    """
    @save_hyperparameters
    def __init__(self,
                 user_vocab_size: int,
                 embedding_size: int
                 ):
        super().__init__()
        self.user_embedding = nn.Embedding(user_vocab_size, embedding_dim=embedding_size)

    def forward(self,
                sequence_representation: SequenceRepresentation
                ) -> ModifiedSequenceRepresentation:
        user = sequence_representation.embedded_elements_sequence.input_sequence.get_attribute(USER_ENTRY_NAME)
        user_emb = self.user_embedding(user).squeeze(1)

        modified_representation = torch.cat([sequence_representation.encoded_sequence, user_emb], 1)
        return ModifiedSequenceRepresentation(modified_representation)
