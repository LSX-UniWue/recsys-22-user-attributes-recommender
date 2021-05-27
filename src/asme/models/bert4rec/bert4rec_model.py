from abc import abstractmethod
from typing import Optional

import torch
from torch import nn

from asme.models.layers.data.sequence import SequenceRepresentation, ModifiedSequenceRepresentation, \
    EmbeddedElementsSequence
from asme.models.layers.layers import build_projection_layer
from asme.models.layers.transformer_layers import TransformerEmbedding, TransformerLayer
from asme.models.sequence_recommendation_model import SequenceRecommenderModel, SequenceRepresentationLayer, \
    SequenceRepresentationModifierLayer, ProjectionLayer
from asme.utils.hyperparameter_utils import save_hyperparameters


class FFNSequenceRepresentationModifierLayer(SequenceRepresentationModifierLayer):
    """
    layer that applies a linear layer and a activation function
    """
    def __init__(self,
                 feature_size: int
                 ):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.GELU(),
            nn.LayerNorm(feature_size)
        )

    def forward(self, sequence_representation: SequenceRepresentation) -> ModifiedSequenceRepresentation:
        transformation = self.transform(sequence_representation.encoded_sequence)
        return ModifiedSequenceRepresentation(sequence_representation.padding_mask,
                                              sequence_representation.attributes,
                                              transformation)


class BidirectionalTransformerSequenceRepresentationLayer(SequenceRepresentationLayer):
    """
    A representation layer that uses a bidirectional transformer layer(s) to encode the given sequence
    """

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 transformer_dropout: float,
                 transformer_attention_dropout: Optional[float] = None,
                 transformer_intermediate_size: Optional[int] = None):
        super().__init__()

        if transformer_intermediate_size is None:
            transformer_intermediate_size = 4 * transformer_hidden_size

        self.transformer_encoder = TransformerLayer(transformer_hidden_size, num_transformer_heads,
                                                    num_transformer_layers, transformer_intermediate_size,
                                                    transformer_dropout,
                                                    attention_dropout=transformer_attention_dropout)

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence
        padding_mask = embedded_sequence.padding_mask

        attention_mask = None

        if padding_mask is not None:
            attention_mask = padding_mask.unsqueeze(1).repeat(1, sequence.size()[1], 1).unsqueeze(1)

        encoded_sequence = self.transformer_encoder(sequence, attention_mask=attention_mask)
        return SequenceRepresentation(padding_mask, embedded_sequence.attributes, encoded_sequence)


class BERT4RecBaseModel(SequenceRecommenderModel):

    @save_hyperparameters
    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 project_layer_type: str = 'transpose_embedding',
                 embedding_pooling_type: str = None,
                 initializer_range: float = 0.02,
                 transformer_intermediate_size: int = None,
                 transformer_attention_dropout: float = None
                 ):
        embedding_layer = self._init_internal(transformer_hidden_size, num_transformer_heads, num_transformer_layers, item_vocab_size,
                                              max_seq_length, transformer_dropout, embedding_pooling_type)

        representation_layer = BidirectionalTransformerSequenceRepresentationLayer(transformer_hidden_size,
                                                                                   num_transformer_heads,
                                                                                   num_transformer_layers,
                                                                                   transformer_dropout,
                                                                                   transformer_attention_dropout,
                                                                                   transformer_intermediate_size)

        projection_layer = self._build_projection_layer(project_layer_type, transformer_hidden_size,
                                                        item_vocab_size)

        transform_layer = FFNSequenceRepresentationModifierLayer(transformer_hidden_size)

        super().__init__(embedding_layer, representation_layer, transform_layer, projection_layer)
        self.initializer_range = initializer_range
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
                       embedding_mode: str = None) -> nn.Module:
        pass

    @abstractmethod
    def _build_projection_layer(self,
                                project_layer_type: str,
                                transformer_hidden_size: int,
                                item_vocab_size: int
                                ) -> ProjectionLayer:
        pass

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
                 initializer_range: float = 0.02,
                 transformer_intermediate_size: int = None,
                 transformer_attention_dropout: float = None
                 ):
        super().__init__(transformer_hidden_size=transformer_hidden_size,
                         num_transformer_heads=num_transformer_heads,
                         num_transformer_layers=num_transformer_layers,
                         item_vocab_size=item_vocab_size,
                         max_seq_length=max_seq_length,
                         transformer_dropout=transformer_dropout,
                         project_layer_type=project_layer_type,
                         embedding_pooling_type=embedding_pooling_type,
                         initializer_range=initializer_range,
                         transformer_intermediate_size=transformer_intermediate_size,
                         transformer_attention_dropout=transformer_attention_dropout)

    def _init_internal(self,
                       transformer_hidden_size: int,
                       num_transformer_heads: int,
                       num_transformer_layers: int,
                       item_vocab_size: int,
                       max_seq_length: int,
                       transformer_dropout: float,
                       embedding_mode: str = None):
        return TransformerEmbedding(item_voc_size=item_vocab_size, max_seq_len=max_seq_length,
                                    embedding_size=transformer_hidden_size, dropout=transformer_dropout,
                                    embedding_pooling_type=embedding_mode)

    def _build_projection_layer(self,
                                project_layer_type: str,
                                transformer_hidden_size: int,
                                item_vocab_size: int
                                ) -> ProjectionLayer:
        return build_projection_layer(project_layer_type, transformer_hidden_size, item_vocab_size,
                                      self.embedding.item_embedding.embedding)

    def _embed_input(self,
                     sequence: torch.Tensor,
                     position_ids: torch.Tensor,
                     **kwargs
                     ) -> torch.Tensor:
        return self.embedding(sequence, position_ids=position_ids)
