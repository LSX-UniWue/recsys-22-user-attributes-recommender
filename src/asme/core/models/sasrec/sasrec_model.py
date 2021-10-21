from asme.core.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer
from asme.core.models.sasrec.components import SASRecProjectionComponent
from asme.core.models.transformer.transformer_encoder_model import TransformerEncoderModel
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class SASRecModel(TransformerEncoderModel):
    
    @save_hyperparameters
    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 embedding_pooling_type: str = None,
                 transformer_intermediate_size: int = None,
                 transformer_attention_dropout: float = None
                 ):
        # We dont set the projection layer immediately since it depends on the embedding, which is not constructed at
        # this point.
        super().__init__(
            transformer_hidden_size=transformer_hidden_size,
            num_transformer_heads=num_transformer_heads,
            num_transformer_layers=num_transformer_layers,
            item_vocab_size=item_vocab_size,
            max_seq_length=max_seq_length,
            transformer_dropout=transformer_dropout,
            bidirectional=False,
            projection_layer=None,
            sequence_representation_modifier_layer=IdentitySequenceRepresentationModifierLayer(),
            embedding_pooling_type=embedding_pooling_type,
            transformer_intermediate_size=transformer_intermediate_size,
            transformer_attention_dropout=transformer_attention_dropout
        )

        self.set_projection_layer(SASRecProjectionComponent(self._sequence_embedding_layer))
