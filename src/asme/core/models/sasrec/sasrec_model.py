from torch import nn

from asme.core.models.common.layers.layers import IdentitySequenceRepresentationModifierLayer
from asme.core.models.common.layers.transformer_layers import TransformerEmbedding
from asme.core.models.sasrec.components import SASRecProjectionComponent
from asme.core.models.transformer.transformer_encoder_model import TransformerEncoderModel
from asme.core.utils.hyperparameter_utils import save_hyperparameters


class SASRecModel(TransformerEncoderModel):
    """
    Implementation of the "Self-Attentive Sequential Recommendation" paper.
    see https://doi.org/10.1109%2fICDM.2018.00035 for more details

    see https://github.com/kang205/SASRec for the original Tensorflow implementation
    """

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

        embedding_layer = TransformerEmbedding(
            item_voc_size=item_vocab_size,
            max_seq_len=max_seq_length,
            embedding_size=transformer_hidden_size,
            dropout=transformer_dropout,
            embedding_pooling_type=embedding_pooling_type
        )

        projection_layer = SASRecProjectionComponent(embedding_layer)

        super().__init__(
            transformer_hidden_size=transformer_hidden_size,
            num_transformer_heads=num_transformer_heads,
            num_transformer_layers=num_transformer_layers,
            transformer_dropout=transformer_dropout,
            bidirectional=False,
            embedding_layer=embedding_layer,
            sequence_representation_modifier_layer=IdentitySequenceRepresentationModifierLayer(),
            projection_layer=projection_layer,
            transformer_intermediate_size=transformer_intermediate_size,
            transformer_attention_dropout=transformer_attention_dropout
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initializes the weights of the layers """
        is_linear_layer = isinstance(module, nn.Linear)
        is_embedding_layer = isinstance(module, nn.Embedding)
        if is_linear_layer or is_embedding_layer:
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if is_linear_layer and module.bias is not None:
            module.bias.data.zero_()
