import functools
from typing import Dict, Any, Optional

from asme.core.models.bert4rec.bert4rec_model import normal_initialize_weights
from asme.core.models.common.components.representation_modifier.ffn_modifier import \
    FFNSequenceRepresentationModifierComponent
from asme.core.models.common.layers.layers import PROJECT_TYPE_LINEAR, build_projection_layer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.kebert4rec.components import KeBERT4RecSequenceElementsRepresentationComponent
from asme.core.models.transformer.transformer_encoder_model import TransformerEncoderModel
from asme.core.utils.hyperparameter_utils import save_hyperparameters
from asme.core.utils.inject import InjectVocabularySize, InjectTokenizers


class KeBERT4RecModel(TransformerEncoderModel):

    @save_hyperparameters
    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: InjectVocabularySize("item"),
                 max_seq_length: int,
                 transformer_dropout: float,
                 additional_attributes: Dict[str, Dict[str, Any]],
                 additional_attributes_tokenizer: InjectTokenizers(),
                 embedding_pooling_type: str = None,
                 initializer_range: float = 0.02,
                 transformer_intermediate_size: Optional[int] = None,
                 transformer_attention_dropout: Optional[float] = None):
        # save for later call by the training module
        self.additional_metadata_keys = list(additional_attributes.keys())

        # embedding will be normed and dropout after all embeddings are added to the representation
        sequence_embedding = TransformerEmbedding(item_vocab_size, max_seq_length, transformer_hidden_size, 0.0,
                                                  embedding_pooling_type=embedding_pooling_type,
                                                  norm_embedding=False)

        element_representation = KeBERT4RecSequenceElementsRepresentationComponent(sequence_embedding,
                                                                                   transformer_hidden_size,
                                                                                   additional_attributes,
                                                                                   additional_attributes_tokenizer,
                                                                                   dropout=transformer_dropout)

        modifier_layer = FFNSequenceRepresentationModifierComponent(transformer_hidden_size)

        projection_layer = build_projection_layer(PROJECT_TYPE_LINEAR, transformer_hidden_size, item_vocab_size,
                                                  sequence_embedding.item_embedding.embedding)

        super().__init__(
            transformer_hidden_size=transformer_hidden_size,
            num_transformer_heads=num_transformer_heads,
            num_transformer_layers=num_transformer_layers,
            transformer_dropout=transformer_dropout,
            embedding_layer=element_representation,
            sequence_representation_modifier_layer=modifier_layer,
            projection_layer=projection_layer,
            bidirectional=True,
            transformer_intermediate_size=transformer_intermediate_size,
            transformer_attention_dropout=transformer_attention_dropout
        )

        # FIXME: move init code
        self.apply(functools.partial(normal_initialize_weights, initializer_range=initializer_range))

    def required_metadata_keys(self):
        return self.additional_metadata_keys
