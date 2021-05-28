from typing import Dict, Any, Optional
from torch import nn

import torch

from asme.models.bert4rec.bert4rec_model import BidirectionalTransformerSequenceRepresentationLayer, \
    FFNSequenceRepresentationModifierLayer
from asme.models.layers.data.sequence import InputSequence, EmbeddedElementsSequence
from asme.models.layers.layers import PROJECT_TYPE_LINEAR
from asme.models.layers.transformer_layers import TransformerEmbedding
from asme.models.sequence_recommendation_model import SequenceElementsRepresentationLayer, SequenceRecommenderModel
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


class KeBERT4RecSequenceElementsRepresentationLayer(SequenceElementsRepresentationLayer):

    def __init__(self,
                 item_vocab_size: int,
                 max_sequence_length: int,
                 embedding_size: int,
                 additional_attributes: Dict[str, Dict[str, Any]],
                 embedding_pooling_type: str = None,
                 ):
        super(KeBERT4RecSequenceElementsRepresentationLayer, self).__init__()

        # embedding will be normed after all embeddings are added to the representation
        self.sequence_embedding = TransformerEmbedding(item_vocab_size, max_sequence_length, embedding_size, 0.0,
                                                       embedding_pooling_type=embedding_pooling_type,
                                                       norm_embedding=False)

        additional_attribute_embeddings = {}
        for attribute_name, attribute_infos in additional_attributes.items():
            embedding_type = attribute_infos['embedding_type']
            vocab_size = attribute_infos['vocab_size']
            additional_attribute_embeddings[attribute_name] = _build_embedding_type(embedding_type=embedding_type,
                                                                                    vocab_size=vocab_size,
                                                                                    hidden_size=embedding_size)
        self.additional_attribute_embeddings = nn.ModuleDict(additional_attribute_embeddings)

        self.dropout_embedding = nn.Dropout(embedding_size)
        self.norm_embedding = nn.LayerNorm(embedding_size)

    def forward(self, sequence: InputSequence) -> EmbeddedElementsSequence:
        position_ids = sequence.get_attribute('position_ids')  # TODO
        embedding = self.sequence_embedding(sequence.sequence, position_ids)
        for input_key, module in self.additional_attribute_embeddings.items():
            embedding += module(sequence.get_attribute(input_key))
        embedding = self.norm_embedding(embedding)
        embedding = self.dropout_embedding(embedding)
        return EmbeddedElementsSequence(embedding)


class KeBERT4RecModel(SequenceRecommenderModel):

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
                 transformer_intermediate_size: Optional[int] = None,
                 transformer_attention_dropout: Optional[float] = None):
        element_representation = KeBERT4RecSequenceElementsRepresentationLayer(item_vocab_size,
                                                                               max_seq_length,
                                                                               transformer_hidden_size,
                                                                               additional_attributes,
                                                                               embedding_pooling_type)
        sequence_representation = BidirectionalTransformerSequenceRepresentationLayer(transformer_hidden_size,
                                                                                      num_transformer_heads,
                                                                                      num_transformer_layers,
                                                                                      transformer_dropout,
                                                                                      transformer_attention_dropout,
                                                                                      transformer_intermediate_size)

        transform_layer = FFNSequenceRepresentationModifierLayer(transformer_hidden_size)

        projection_layer = self._build_projection_layer(PROJECT_TYPE_LINEAR, transformer_hidden_size,
                                                        item_vocab_size)

        super().__init__(element_representation, sequence_representation, transform_layer, projection_layer)

        # FIXME:
        self.initializer_range = initializer_range
        self.apply(self._init_weights)
