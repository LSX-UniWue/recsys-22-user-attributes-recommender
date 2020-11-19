from abc import abstractmethod

import math
import torch

from torch import nn

from configs.models.bert4rec.bert4rec_config import BERT4RecConfig
from models.layers.transformer_layers import TransformerEmbedding


class BERT4RecBaseModel(nn.Module):

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 transformer_dropout: float,
                 ):
        super().__init__()

        self.transformer_hidden_size = transformer_hidden_size
        self.num_transformer_heads = num_transformer_heads
        self.num_transformer_layers = num_transformer_layers
        self.transformer_dropout = transformer_dropout

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.transformer_hidden_size,
                                                    nhead=self.num_transformer_heads,
                                                    dim_feedforward=self.transformer_hidden_size,
                                                    dropout=self.transformer_dropout,
                                                    activation='gelu')

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers,
                                                         num_layers=self.num_transformer_heads)

        # for decoding the sequence into the item space again
        self.linear = nn.Linear(self.transformer_hidden_size, self.transformer_hidden_size)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(self.transformer_hidden_size)

    def forward(self,
                input_seq: torch.Tensor,
                position_ids: torch.Tensor = None,
                padding_mask: torch.Tensor = None,
                **kwargs
                ) -> torch.Tensor:
        """
        forward pass to calculate the scores for the mask item modelling

        :param input_seq: the input sequence :math:`(S, N)`
        :param position_ids: (optional) positional_ids if None the position ids are generated :math:`(S, N)`
        :param padding_mask: (optional) the padding mask if the sequence is padded :math:`(N, S)`
        :return: the logits of the predicted tokens :math:`(S, N, I)`
        (Note: all logits for all positions are returned. For loss calculation please only use the positions of the
        MASK tokens.)

        Where S is the (max) sequence length of the batch, N the batch size, and I the vocabulary size of the items.
        """
        # H is the transformer hidden size
        # embed the input
        input_seq = self._embed_input(input_sequence=input_seq,
                                      position_ids=position_ids,
                                      kwargs=kwargs)  # (S, N, H)

        # use the bidirectional transformer
        input_seq = self.transformer_encoder(src=input_seq,
                                             src_key_padding_mask=padding_mask)  # (S, N, H)

        # decode the hidden representation
        dense = self.gelu(self.linear(input_seq))  # (S, N, H)

        # norm the output
        dense = self.layer_norm(dense)  # (S, N, H)

        return self._projection(dense)

    @abstractmethod
    def _embed_input(self,
                     input_sequence: torch.Tensor,
                     position_ids: torch.Tensor,
                     **kwargs
                     ) -> torch.Tensor:
        pass

    @abstractmethod
    def _projection(self,
                    dense: torch.Tensor
                    ) -> torch.Tensor:
        pass


class BERT4RecModel(BERT4RecBaseModel):
    """
        implementation of the paper "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations
        from Transformer"
        see https://doi.org/10.1145%2f3357384.3357895 for more details.
    """

    @classmethod
    def from_config(cls, config: BERT4RecConfig):
        return cls(transformer_hidden_size=config.transformer_hidden_size,
                   num_transformer_heads=config.num_transformer_heads,
                   num_transformer_layers=config.num_transformer_layers,
                   item_vocab_size=config.item_vocab_size,
                   max_seq_length=config.max_seq_length,
                   transformer_dropout=config.transformer_dropout)

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 initializer_range: float = 0.02,
                 embedding_mode: str = None
                 ):
        super().__init__(transformer_hidden_size=transformer_hidden_size,
                         num_transformer_heads=num_transformer_heads,
                         num_transformer_layers=num_transformer_layers,
                         transformer_dropout=transformer_dropout)

        self.item_vocab_size = item_vocab_size
        self.max_seq_length = max_seq_length + 1
        self.embedding_mode = embedding_mode
        self.initializer_range = initializer_range

        self.embedding = TransformerEmbedding(item_voc_size=self.item_vocab_size,
                                              max_seq_len=self.max_seq_length,
                                              embedding_size=self.transformer_hidden_size,
                                              dropout=self.transformer_dropout,
                                              embedding_mode=self.embedding_mode,
                                              initializer_range=self.initializer_range)

        self.output_bias = nn.Parameter(torch.Tensor(self.item_vocab_size))

        self.init_weights()

    def init_weights(self):
        bound = 1 / math.sqrt(self.item_vocab_size)
        nn.init.uniform_(self.output_bias, -bound, bound)

    def _embed_input(self,
                     input_sequence: torch.Tensor,
                     position_ids: torch.Tensor,
                     **kwargs
                     ) -> torch.Tensor:
        return self.embedding(input_sequence=input_sequence,
                              position_ids=position_ids)

    def _projection(self,
                    dense: torch.Tensor
                    ) -> torch.Tensor:
        dense = torch.matmul(dense, self.embedding.get_item_embedding_weight().transpose(0, 1))  # (S, N, I)
        return dense + self.output_bias
