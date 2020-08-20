import torch
import torch.nn as nn

from models.layers.util_layers import MatrixFactorizationLayer, TransformerEmbedding
from config.sasrec.sas_rec_config import SasConfig
from utils.tensor_utils import generate_square_subsequent_mask


class SasRecModel(nn.Module):
    """
    Implementation of the "Self-Attentive Sequential Recommendation" paper.
    see https://doi.org/10.1109%2fICDM.2018.00035 for more details
    """

    def __init__(self, config: SasConfig):
        super().__init__()
        self.d_model = config.d_model

        self.embedding = TransformerEmbedding(config.item_voc_size, config.max_seq_length, self.d_model)

        encoder_layers = nn.TransformerEncoderLayer(self.d_model, config.num_transformer_heads, self.d_model,
                                                    config.transformer_dropout)
        # TODO: check add norm?
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.num_transformer_layers)

        self.input_sequence_mask = None
        self.mfl = MatrixFactorizationLayer(self.embedding.get_item_embedding_weight())
        self.softmax = nn.Softmax(dim=2)

    def forward(self,
                input_sequence: torch.Tensor,
                position_ids: torch.Tensor = None,
                padding_mask: torch.Tensor = None):
        """
        forword pass to generate the scores for the next items in the sequence

        :param input_sequence: the sequence [S x B]
        :param position_ids: the optional position ids [S x B] if not the position ids are generated
        :param padding_mask: the optional padding mask [B x S] if the sequence is padded
        :return: the scores for each sequence step [S x B x I]

        Where S is the (max) sequence length of the batch, B the batch size, and I the vocabulary size of the items.
        """
        device = input_sequence.device
        if self.input_sequence_mask is None or self.input_sequence_mask.size(0) != len(input_sequence):
            # to mask the next words we generate a triangle mask
            mask = generate_square_subsequent_mask(len(input_sequence)).to(device)
            self.input_sequence_mask = mask

        input_sequence = self.embedding(input_sequence, position_ids)

        transformer_output = self.transformer_encoder(input_sequence,
                                                      mask=self.input_sequence_mask,
                                                      src_key_padding_mask=padding_mask)

        scores = self.mfl(transformer_output)
        scores = self.softmax(scores)

        return scores
