from typing import Dict
from torch import nn

import torch

from models.bert4rec.bert4rec_model import BERT4RecBaseModel, BERT4REC_PROJECT_TYPE_LINEAR
from models.layers.transformer_layers import TransformerEmbedding


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


class KeBERT4Rec(BERT4RecBaseModel):

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 additional_attributes: Dict[str, (str, int)],
                 embedding_pooling_type: str = None):
        super().__init__(transformer_hidden_size=transformer_hidden_size,
                         num_transformer_heads=num_transformer_heads,
                         num_transformer_layers=num_transformer_layers,
                         transformer_dropout=transformer_dropout,
                         item_vocab_size=item_vocab_size,
                         max_seq_length=max_seq_length,
                         project_layer_type=BERT4REC_PROJECT_TYPE_LINEAR,
                         embedding_pooling_type=embedding_pooling_type)

        self.item_vocab_size = item_vocab_size
        self.max_seq_length = max_seq_length + 1
        self.additional_attributes = additional_attributes

        # here we set the dropout to 0 and norm to false because we want to norm and drop after applying
        # the additional embeddings
        self.embedding = TransformerEmbedding(item_voc_size=self.item_vocab_size,
                                              max_seq_len=self.max_seq_length,
                                              embedding_size=self.transformer_hidden_size,
                                              dropout=0.0,
                                              norm_embedding=False)

        # build all additional attribute embeddings
        additional_attribute_embeddings = {}
        for key, (embedding_type, vocab_size) in self.additional_attributes.items():
            additional_attribute_embeddings[key] = _build_embedding_type(embedding_type=embedding_type,
                                                                         vocab_size=vocab_size,
                                                                         hidden_size=transformer_hidden_size)

        self.additional_attribute_embeddings = nn.ModuleDict(additional_attribute_embeddings)

        self.embedding_norm = nn.LayerNorm(self.transformer_hidden_size)
        self.dropout_layer = nn.Dropout(p=self.transformer_dropout)

    def _embed_input(self,
                     input_sequence: torch.Tensor,
                     position_ids: torch.Tensor,
                     **kwargs
                     ) -> torch.Tensor:
        embedding = self.embedding(input_sequence, position_ids)
        for input_key, module in self.additional_attribute_embeddings:
            embedding += module(kwargs[input_key])
        embedding = self.dropout_layer(embedding)
        return embedding
