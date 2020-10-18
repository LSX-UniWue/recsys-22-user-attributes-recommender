import torch

from torch import nn

from configs.models.bert4rec.bert4rec_config import BERT4RecConfig
from models.layers.transformer_layers import TransformerEmbedding


class BERT4RecModel(nn.Module):

    @classmethod
    def from_config(cls, config: BERT4RecConfig):

        return cls(transformer_hidden_size=config.transformer_hidden_size,
                   num_transformer_heads=config.num_transformer_heads,
                   num_transformer_layers=config.num_transformer_layers,
                   item_vocab_size=config.item_voc_size,
                   max_seq_length=config.max_seq_length,
                   dropout=config.transformer_dropout)
    """
    implementation of the paper "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"
    see https://doi.org/10.1145%2f3357384.3357895 for more details.
    """

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 dropout: float):
        super().__init__()

        self.transformer_hidden_size = transformer_hidden_size
        self.num_transformer_heads = num_transformer_heads
        self.num_transformer_layers = num_transformer_layers
        self.item_vocab_size = item_vocab_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout

        self.embedding = TransformerEmbedding(item_voc_size=self.item_vocab_size,
                                              max_seq_len=self.max_seq_length,
                                              embedding_size=self.transformer_hidden_size,
                                              dropout=self.dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.transformer_hidden_size,
                                                    nhead = self.num_transformer_heads,
                                                    dim_feedforward=self.transformer_hidden_size,
                                                    dropout=self.dropout,
                                                    activation='gelu')

        # TODO: check encoder norm?
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers,
                                                         num_layers=self.num_transformer_heads)

        # for decoding the sequence into the item space again
        # TODO: init the linear layer:
        # kernel_initializer=modeling.create_initializer(
        #                         bert_config.initializer_range))
        self.linear = nn.Linear(self.transformer_hidden_size, self.transformer_hidden_size)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(self.transformer_hidden_size)
        self.output_bias = nn.Parameter(torch.rand(self.item_vocab_size))

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
        # H is the transformer hidden size
        # embed the input
        input_seq = self.embedding(input_seq, position_ids)  # (S, N, H)

        # use the bidirectional transformer
        input_seq = self.transformer_encoder(input_seq,
                                             src_key_padding_mask=padding_mask)  # (S, N, H)
        # decode the hidden representation
        dense = self.gelu(self.linear(input_seq))  # (S, N, H)

        # norm the output
        dense = self.layer_norm(dense)  # (S, N, H)

        dense = torch.matmul(dense, self.embedding.get_item_embedding_weight().transpose(0, 1))
        dense = dense + self.output_bias

        return dense
