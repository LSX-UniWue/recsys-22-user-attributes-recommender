import torch

from torch import nn

from configs.models.bert4rec.bert4rec_config import BERT4RecConfig
from models.layers.util_layers import MatrixFactorizationLayer
from models.layers.transformer_layers import TransformerEmbedding

CROSS_ENTROPY_IGNORE_INDEX = -100


class BERT4RecModel(nn.Module):
    """
    implementation of the paper "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"
    see https://doi.org/10.1145%2f3357384.3357895 for more details.
    """

    def __init__(self,
                 config: BERT4RecConfig):
        super().__init__()

        self.config = config

        d_model = config.transformer_hidden_size
        dropout = config.transformer_dropout
        self.embedding = TransformerEmbedding(config.item_voc_size, config.max_seq_length, d_model, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model, config.num_transformer_heads, d_model,
                                                    dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.num_transformer_layers)

        # for decoding the sequence into the item space again
        self.linear = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        self.mfl = MatrixFactorizationLayer(self.embedding.get_item_embedding_weight())
        self.softmax = nn.Softmax(dim=2)

    def forward(self,
                input_seq: torch.Tensor,
                position_ids: torch.Tensor = None,
                padding_mask: torch.Tensor = None):
        """
        forward pass to calculate the scores for the mask item modelling

        :param input_seq: the input sequence [S x B]
        :param position_ids: (optional) positional_ids if None the position ids are generated [S x B]
        :param padding_mask: (optional) the padding mask if the sequence is padded [B x S]
        :return: the scores of the predicted tokens [S x B x I] (Note: here all scores for all positions are returned.
        For loss calculation please only use MASK tokens.)

        Where S is the (max) sequence length of the batch, B the batch size, and I the vocabulary size of the items.
        """
        # embed the input
        input_seq = self.embedding(input_seq, position_ids)

        # use the bidirectional transformer
        input_seq = self.transformer_encoder(input_seq,
                                             src_key_padding_mask=padding_mask)

        # decode the hidden representation
        scores = self.gelu(self.linear(input_seq))

        scores = self.mfl(scores)
        scores = self.softmax(scores)
        return scores
