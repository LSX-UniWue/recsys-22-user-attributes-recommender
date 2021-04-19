from typing import Dict, Any
from torch import nn

import torch

from asme.models.bert4rec.bert4rec_model import BERT4RecBaseModel
from asme.models.layers.layers import PROJECT_TYPE_LINEAR
from asme.models.layers.layers import build_projection_layer
from asme.models.layers.transformer_layers import TransformerEmbedding
from asme.utils.hyperparameter_utils import save_hyperparameters


def _build_embedding_type(embedding_type: str,
                          vocab_size: int,
                          hidden_size: int
                          ) -> nn.Module:
    return {
        'content_embedding': nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=hidden_size),
        'linear_upscale': LinearUpscaler(vocab_size=vocab_size,
                                         embed_size=hidden_size)
    }[embedding_type]


class LinearUpscaler(nn.Module):

    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.linear = nn.Linear(vocab_size, embed_size)
        self.vocab_size = vocab_size

    def forward(self, content_input):
        # the input is a sequence of content ids without any order
        # so we convert them into a multi-hot encoding
        multi_hot = torch.nn.functional.one_hot(content_input, self.vocab_size).sum(2).float()
        # 0 is the padding category, so zero it out
        multi_hot[:, :, 0] = 0
        return self.linear(multi_hot)


class KeBERT4RecModel(BERT4RecBaseModel):

    @save_hyperparameters
    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 additional_attributes: Dict[str, Dict[str, Any]],
                 embedding_pooling_type: str = None,
                 initializer_range: float = 0.02,
                 transformer_intermediate_size: int = None,
                 transformer_attention_dropout: float = None):

        super().__init__(transformer_hidden_size=transformer_hidden_size,
                         num_transformer_heads=num_transformer_heads,
                         num_transformer_layers=num_transformer_layers,
                         transformer_dropout=transformer_dropout,
                         item_vocab_size=item_vocab_size,
                         max_seq_length=max_seq_length,
                         project_layer_type=PROJECT_TYPE_LINEAR,
                         embedding_pooling_type=embedding_pooling_type,
                         initializer_range=initializer_range,
                         transformer_intermediate_size=transformer_intermediate_size,
                         transformer_attention_dropout=transformer_attention_dropout)

        self.additional_attributes = additional_attributes
        additional_attribute_embeddings = {}
        for attribute_name, attribute_infos in additional_attributes.items():
            embedding_type = attribute_infos['embedding_type']
            vocab_size = attribute_infos['vocab_size']
            additional_attribute_embeddings[attribute_name] = _build_embedding_type(embedding_type=embedding_type,
                                                                                    vocab_size=vocab_size,
                                                                                    hidden_size=transformer_hidden_size)
        self.additional_attribute_embeddings = nn.ModuleDict(additional_attribute_embeddings)
        self.dropout_embedding = nn.Dropout(transformer_dropout)
        self.norm_embedding = nn.LayerNorm(transformer_hidden_size)
        self.apply(self._init_weights)

    def _init_internal(self,
                       transformer_hidden_size: int,
                       num_transformer_heads: int,
                       num_transformer_layers: int,
                       item_vocab_size: int,
                       max_seq_length: int,
                       transformer_dropout: float,
                       embedding_mode: str = None):
        # we here do not norm the embedding, we will do it after all attribute embeddings were added
        # to the global embedding
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
        embedding = self.embedding(sequence, position_ids)
        for input_key, module in self.additional_attribute_embeddings.items():
            embedding += module(kwargs[input_key])
        embedding = self.norm_embedding(embedding)
        embedding = self.dropout_embedding(embedding)
        return embedding
