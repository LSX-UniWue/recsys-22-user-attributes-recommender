import math
from abc import abstractmethod

import torch
from torch import nn

import torch.nn.functional as F

from models.layers.transformer_layers import TransformerEmbedding

BERT4REC_PROJECT_TYPE_LINEAR = 'linear'


class BERT4RecProjectionLayer(nn.Module):

    @abstractmethod
    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        pass


def _build_projection_layer(project_type: str,
                            transformer_hidden_size: int,
                            item_voc_size: int,
                            embedding: TransformerEmbedding
                            ) -> BERT4RecProjectionLayer:
    if project_type == BERT4REC_PROJECT_TYPE_LINEAR:
        return BERT4RecLinearProjectionLayer(transformer_hidden_size, item_voc_size)

    if project_type == 'transpose_embedding':
        return BERT4RecItemEmbeddingProjectionLayer(item_voc_size, embedding)

    raise KeyError(f'{project_type} invalid projection layer')


class BERT4RecLinearProjectionLayer(BERT4RecProjectionLayer):

    def __init__(self,
                 transformer_hidden_size: int,
                 item_vocab_size: int):
        super().__init__()

        self.linear = nn.Linear(transformer_hidden_size, item_vocab_size)

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        return self.linear(dense)


class BERT4RecItemEmbeddingProjectionLayer(BERT4RecProjectionLayer):

    def __init__(self,
                 item_vocab_size: int,
                 embedding: TransformerEmbedding
                 ):
        super().__init__()

        self.item_vocab_size = item_vocab_size

        self.embedding = embedding
        self.output_bias = nn.Parameter(torch.Tensor(self.item_vocab_size))

        self.init_weights()

    def init_weights(self):
        bound = 1 / math.sqrt(self.item_vocab_size)
        nn.init.uniform_(self.output_bias, -bound, bound)

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        dense = torch.matmul(dense, self.embedding.get_item_embedding_weight().transpose(0, 1))  # (S, N, I)
        return dense + self.output_bias


class BERT4RecBaseModel(nn.Module):

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 project_layer_type: str = 'transpose_embedding',
                 embedding_mode: str = None
                 ):
        super().__init__()

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(transformer_hidden_size, num_transformer_heads, transformer_hidden_size * 4, transformer_dropout) for _ in range(num_transformer_layers)])

        self.transform = nn.Linear(transformer_hidden_size, transformer_hidden_size)
        self.gelu = nn.GELU()

        self._init_internal(transformer_hidden_size, num_transformer_heads, num_transformer_layers, item_vocab_size,
                            max_seq_length, transformer_dropout, embedding_mode)

        self.projection_layer = self._build_projection_layer(project_layer_type, transformer_hidden_size,
                                                             item_vocab_size)

    def _init_internal(self,
                       transformer_hidden_size: int,
                       num_transformer_heads: int,
                       num_transformer_layers: int,
                       item_vocab_size: int,
                       max_seq_length: int,
                       transformer_dropout: float,
                       embedding_mode: str = None):
        pass

    @abstractmethod
    def _build_projection_layer(self,
                                project_layer_type: str,
                                transformer_hidden_size: int,
                                item_vocab_size: int
                                ) -> nn.Module:
        pass

    def forward(self,
                sequence: torch.Tensor,
                position_ids: torch.Tensor = None,
                padding_mask: torch.Tensor = None,
                **kwargs
                ) -> torch.Tensor:

        # embedding the indexed sequence to sequence of vectors
        encoded_sequence = self.embedding(sequence, position_ids=position_ids)

        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).repeat(1, sequence.size(1), 1).unsqueeze(1)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            encoded_sequence = transformer.forward(encoded_sequence, padding_mask)

        transformed = self.gelu(self.transform(encoded_sequence))

        return self.projection_layer(transformed)

    @abstractmethod
    def _embed_input(self,
                     sequence: torch.Tensor,
                     position_ids: torch.Tensor,
                     **kwargs
                     ) -> torch.Tensor:
        pass


class BERT4RecModel(BERT4RecBaseModel):

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 project_layer_type: str = 'transpose_embedding',
                 embedding_mode: str = None
                 ):
        super().__init__(transformer_hidden_size=transformer_hidden_size,
                         num_transformer_heads=num_transformer_heads,
                         num_transformer_layers=num_transformer_layers,
                         item_vocab_size=item_vocab_size,
                         max_seq_length=max_seq_length,
                         transformer_dropout=transformer_dropout,
                         project_layer_type=project_layer_type,
                         embedding_mode=embedding_mode)

    def _init_internal(self,
                       transformer_hidden_size: int,
                       num_transformer_heads: int,
                       num_transformer_layers: int,
                       item_vocab_size: int,
                       max_seq_length: int,
                       transformer_dropout: float,
                       embedding_mode: str = None):
        max_seq_length = max_seq_length + 1
        self.embedding = TransformerEmbedding(item_voc_size=item_vocab_size, max_seq_len=max_seq_length,
                                              embedding_size=transformer_hidden_size, dropout=transformer_dropout,
                                              embedding_mode=embedding_mode, norm_embedding=False)

    def _build_projection_layer(self,
                                project_layer_type: str,
                                transformer_hidden_size: int,
                                item_vocab_size: int
                                ) -> nn.Module:
        return _build_projection_layer(project_layer_type, transformer_hidden_size, item_vocab_size,
                                       self.embedding)

    def _embed_input(self,
                     sequence: torch.Tensor,
                     position_ids: torch.Tensor,
                     **kwargs
                     ) -> torch.Tensor:
        return self.embedding(sequence, position_ids=position_ids)



class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = nn.Embedding(vocab_size, embed_size)
        self.position = nn.Embedding(max_len, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        batch_size = sequence.size(0)
        position_embedding = self.position.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        x = self.token(sequence) + position_embedding
        return self.dropout(x)


class LayerNorm(nn.Module):
    """
        Construct a layernorm module (See citation for details)."
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class Attention(nn.Module):
    """
    Computes 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self,
                 h: int,
                 d_model: int,
                 dropout: float = 0.1
                 ):
        super().__init__()
        assert d_model % h == 0

        # we assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = None
                ) -> torch.Tensor:
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self,
                 hidden: int,
                 attn_heads: int,
                 feed_forward_hidden: int,
                 dropout: float):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                input_sequence: torch.Tensor,
                mask: torch.Tensor
                ) -> torch.Tensor:
        input_sequence = self.input_sublayer(input_sequence, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        input_sequence = self.output_sublayer(input_sequence, self.feed_forward)
        return self.dropout(input_sequence)
