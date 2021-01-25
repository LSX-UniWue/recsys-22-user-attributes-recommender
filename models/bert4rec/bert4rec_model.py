from abc import abstractmethod

import torch
from torch import nn

from models.layers.layers import build_projection_layer
from models.layers.transformer_layers import TransformerEmbedding, TransformerLayer


class BERT4RecBaseModel(nn.Module):

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 project_layer_type: str = 'transpose_embedding',
                 embedding_pooling_type: str = None,
                 initializer_range: float = 0.02
                 ):
        super().__init__()
        self.initializer_range = initializer_range
        self.transformer_encoder = TransformerLayer(transformer_hidden_size, num_transformer_heads,
                                                    num_transformer_layers, transformer_hidden_size * 4,
                                                    transformer_dropout)

        self.transform = nn.Sequential(
            nn.Linear(transformer_hidden_size, transformer_hidden_size),
            nn.GELU(),
            nn.LayerNorm(transformer_hidden_size)
        )

        self._init_internal(transformer_hidden_size, num_transformer_heads, num_transformer_layers, item_vocab_size,
                            max_seq_length, transformer_dropout, embedding_pooling_type)

        self.projection_layer = self._build_projection_layer(project_layer_type, transformer_hidden_size,
                                                             item_vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initializes the weights of the layers """
        is_linear_layer = isinstance(module, nn.Linear)
        is_embedding_layer = isinstance(module, nn.Embedding)
        if is_linear_layer or is_embedding_layer:
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if is_linear_layer and module.bias is not None:
            module.bias.data.zero_()

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

        transformed = self.transform(encoded_sequence)

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
                 embedding_pooling_type: str = None,
                 initializer_range: float = 0.02
                 ):
        super().__init__(transformer_hidden_size=transformer_hidden_size,
                         num_transformer_heads=num_transformer_heads,
                         num_transformer_layers=num_transformer_layers,
                         item_vocab_size=item_vocab_size,
                         max_seq_length=max_seq_length,
                         transformer_dropout=transformer_dropout,
                         project_layer_type=project_layer_type,
                         embedding_pooling_type=embedding_pooling_type,
                         initializer_range=initializer_range)

    def _init_internal(self,
                       transformer_hidden_size: int,
                       num_transformer_heads: int,
                       num_transformer_layers: int,
                       item_vocab_size: int,
                       max_seq_length: int,
                       transformer_dropout: float,
                       embedding_mode: str = None):
        self.embedding = TransformerEmbedding(item_voc_size=item_vocab_size, max_seq_len=max_seq_length,
                                              embedding_size=transformer_hidden_size, dropout=transformer_dropout,
                                              embedding_pooling_type=embedding_mode, norm_embedding=False)

    def _build_projection_layer(self,
                                project_layer_type: str,
                                transformer_hidden_size: int,
                                item_vocab_size: int
                                ) -> nn.Module:
        return build_projection_layer(project_layer_type, transformer_hidden_size, item_vocab_size,
                                      self.embedding.item_embedding.embedding)

    def _embed_input(self,
                     sequence: torch.Tensor,
                     position_ids: torch.Tensor,
                     **kwargs
                     ) -> torch.Tensor:
        return self.embedding(sequence, position_ids=position_ids)
