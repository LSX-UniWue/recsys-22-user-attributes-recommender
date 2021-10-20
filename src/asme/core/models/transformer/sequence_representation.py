from typing import Optional

import torch

from asme.core.models.common.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation
from asme.core.models.common.layers.layers import SequenceRepresentationLayer
from asme.core.models.common.layers.transformer_layers import TransformerLayer


class TransformerSequenceRepresentationComponent(SequenceRepresentationLayer):

    def __init__(self, transformer_layer: TransformerLayer, bidirectional: bool):
        super().__init__()
        self.transformer_layer = transformer_layer
        self.bidirectional = bidirectional

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        sequence = embedded_sequence.embedded_sequence
        padding_mask = embedded_sequence.input_sequence.padding_mask

        input_size = sequence.size()
        batch_size = input_size[0]
        sequence_length = input_size[1]

        """ 
        We have to distinguish 4 cases here:
            - Bidirectional and no padding mask: Transformer can attend to all tokens with no restrictions
            - Bidirectional and padding mask: Transformer can attend to all tokens but those marked with the padding 
              mask
            - Unidirectional and no padding mask: Transformer can attend to all tokens up to the current sequence index
            - Unidirectional and padding mask: Transformer can attend to all tokens up to the current sequence index
              except those marked by the padding mask
        """
        if self.bidirectional:
            if padding_mask is None:
                attention_mask = None
            else:
                attention_mask = padding_mask.unsqueeze(1).repeat(1, sequence_length, 1).unsqueeze(1)
        else:
            if padding_mask is None:
                attention_mask = torch.tril(torch.ones([sequence_length, sequence_length],device=sequence.device)).unsqueeze(0).repeat(batch_size,1,1)
            else:
                attention_mask = torch.tril(torch.ones([sequence_length, sequence_length],device=sequence.device)).unsqueeze(0).repeat(batch_size,1,1)
                attention_mask *= padding_mask.unsqueeze(1).repeat(1, sequence_length, 1)

        encoded_sequence = self.transformer_encoder(sequence, attention_mask=attention_mask)
        return SequenceRepresentation(encoded_sequence)


class UnidirectionalTransformerSequenceRepresentationComponent(SequenceRepresentationLayer):

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
        representation = self.transformer_layer(sequence, attention_mask=attention_mask)

        return SequenceRepresentation(representation)


class BidirectionalTransformerSequenceRepresentationComponent(SequenceRepresentationLayer):
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
        padding_mask = embedded_sequence.input_sequence.padding_mask

        attention_mask = None

        if padding_mask is not None:
            attention_mask = padding_mask.unsqueeze(1).repeat(1, sequence.size()[1], 1).unsqueeze(1)

        encoded_sequence = self.transformer_encoder(sequence, attention_mask=attention_mask)
        return SequenceRepresentation(encoded_sequence)