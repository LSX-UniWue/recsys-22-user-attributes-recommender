from typing import Union, Tuple

import torch
import torch.nn as nn

from asme.models.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation, \
    ModifiedSequenceRepresentation
from asme.models.layers.layers import SequenceRepresentationLayer, ProjectionLayer, \
    IdentitySequenceRepresentationModifierLayer
from asme.models.layers.transformer_layers import TransformerEmbedding, TransformerLayer
from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.utils.hyperparameter_utils import save_hyperparameters


class SASRecTransformerLayer(SequenceRepresentationLayer):

    def __init__(self, transformer_layer: TransformerLayer):
        super().__init__()
        self.transformer_layer = transformer_layer

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:

        sequence = embedded_sequence.embedded_sequence

        # pipe the embedded sequence to the transformer
        # first build the attention mask
        input_size = sequence.size()
        batch_size = input_size[0]
        sequence_length = input_size[1]

        attention_mask = torch.triu(torch.ones([sequence_length, sequence_length], device=sequence.device)) \
            .transpose(1, 0).unsqueeze(0).repeat(batch_size, 1, 1)

        if embedded_sequence.input_sequence.has_attribute("padding_mask"):
            padding_mask = embedded_sequence.input_sequence.get_attribute("padding_mask")
            attention_mask = attention_mask * padding_mask.unsqueeze(1).repeat(1, sequence_length, 1)

        attention_mask = attention_mask.unsqueeze(1).to(dtype=torch.bool)

        return self.transformer_layer(embedded_sequence, attention_mask=attention_mask)


class SASRecProjectionLayer(ProjectionLayer):

    def __init__(self, embedding: TransformerEmbedding):
        super().__init__()
        self.embedding = embedding

    def forward(self,
                modified_sequence_representation: ModifiedSequenceRepresentation
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        input_sequence = modified_sequence_representation.input_sequence

        positive_samples = input_sequence.get_attribute("positive_samples")
        negative_samples = input_sequence.get_attribute("negative_samples")

        padding_mask = input_sequence.padding_mask
        transformer_output = modified_sequence_representation.modified_encoded_sequence
        # TODO (AD) We should switch between training / prediction based on Module state, see: torch.nn.Module/train
        # when training the model we multiply the seq embedding with the positive and negative items
        if negative_samples is not None:
            emb_pos_items = self.embedding.get_item_embedding(positive_samples)  # (N, H)
            emb_neg_items = self.embedding.get_item_embedding(negative_samples)  # (N, H)

            pos_output = emb_pos_items * transformer_output  # (N, S, H)
            neg_output = emb_neg_items * transformer_output  # (N, S, H)

            pos_output = torch.sum(pos_output, -1)  # (N, S)
            neg_output = torch.sum(neg_output, -1)  # (N, S)

            return pos_output, neg_output

        # inference step (I is the number of positive items to test)
        # embeddings of pos_items
        item_embeddings = self.embedding.get_item_embedding(positive_samples, flatten=False)  # (N, I, H)

        # we use "advanced" indexing to slice the right elements from the transformer output
        batch_size = transformer_output.size()[0]
        batch_index = torch.arange(0, batch_size)

        # calculate indices from the padding mask
        seq_index = padding_mask.sum(-1) - 1
        transformer_last_pos_output = transformer_output[batch_index, seq_index]  # (N, H)

        # now matmul it with the item embeddings
        logits = item_embeddings.matmul(transformer_last_pos_output.unsqueeze(-1))

        return logits.squeeze(-1)


class SASRecModel(SequenceRecommenderModel):
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
        """
        inits the SASRec model
        :param transformer_hidden_size: the hidden size of the transformer
        :param num_transformer_heads: the number of heads of the transformer
        :param num_transformer_layers: the number of layers of the transformer
        :param item_vocab_size: the item vocab size
        :param max_seq_length: the max sequence length
        :param transformer_dropout: the dropout of the model
        :param embedding_pooling_type: the pooling to use for basket recommendation
        :param transformer_intermediate_size: the intermediate size of the transformer (default 4 * transformer_hidden_size)
        :param transformer_attention_dropout: the attention dropout (default transformer_dropout)
        """

        if transformer_intermediate_size is None:
            transformer_intermediate_size = 4 * transformer_hidden_size

        embedding_layer = TransformerEmbedding(item_voc_size=item_vocab_size,
                                               max_seq_len=max_seq_length,
                                               embedding_size=transformer_hidden_size,
                                               dropout=transformer_dropout,
                                               embedding_pooling_type=embedding_pooling_type)

        transformer_layer = TransformerLayer(transformer_hidden_size,
                                             num_transformer_heads,
                                             num_transformer_layers,
                                             transformer_intermediate_size,
                                             transformer_dropout,
                                             attention_dropout=transformer_attention_dropout)
        sasrec_transformer_layer = SASRecTransformerLayer(transformer_layer)

        modified_seq_representation_layer = IdentitySequenceRepresentationModifierLayer()
        sasrec_projection_layer = SASRecProjectionLayer(embedding_layer)

        super().__init__(sequence_embedding_layer=embedding_layer,
                         sequence_representation_layer=sasrec_transformer_layer,
                         sequence_representation_modifier_layer=modified_seq_representation_layer,
                         projection_layer=sasrec_projection_layer)

        # FIXME (AD) I think we should move this out of the model and call it through a callback before training starts
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
