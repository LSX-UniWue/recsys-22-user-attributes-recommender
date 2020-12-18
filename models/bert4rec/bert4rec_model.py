import math
from abc import abstractmethod

import torch
from torch import nn

from models.layers.transformer_layers import TransformerEmbedding, TransformerLayer

BERT4REC_PROJECT_TYPE_LINEAR = 'linear'


class BERT4RecProjectionLayer(nn.Module):

    @abstractmethod
    def forward(self,
                dense: torch.Tensor
                ) -> torch.Tensor:
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

        self.transformer_encoder = TransformerLayer(transformer_hidden_size, num_transformer_heads,
                                                    num_transformer_layers, transformer_hidden_size * 4,
                                                    transformer_dropout)

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
        """
        forward pass to calculate the scores for the mask item modelling

        :param sequence: the input sequence :math:`(N, S)`
        :param position_ids: (optional) positional_ids if None the position ids are generated :math:`(N, S)`
        :param padding_mask: (optional) the padding mask if the sequence is padded :math:`(N, S)`
        :return: the logits of the predicted tokens :math:`(N, S, I)`
        (Note: all logits for all positions are returned. For loss calculation please only use the positions of the
        MASK tokens.)

        Where N the batch size, S is the (max) sequence length of the batch, and I the vocabulary size of the items.
        """

        # embedding the indexed sequence to sequence of vectors
        embedded_sequence = self.embedding(sequence, position_ids=position_ids)

        encoded_sequence = self.transformer_encoder(embedded_sequence, padding_mask=padding_mask)

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
    """
        implementation of the paper "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations
        from Transformer"
        see https://doi.org/10.1145%2f3357384.3357895 for more details.
        Using own transformer implementation to be able to pass batch first tensors to the model
    """

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
