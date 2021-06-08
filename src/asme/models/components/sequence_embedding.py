from typing import Optional

from asme.models.layers.data.sequence import InputSequence, EmbeddedElementsSequence
from asme.models.layers.layers import SequenceElementsRepresentationLayer
from asme.models.layers.sequence_embedding import SequenceElementsEmbeddingLayer


class SequenceElementsEmbeddingComponent(SequenceElementsRepresentationLayer):
    """
    A component that projects every element in a sequence into an embedding space. The component also supports pooling
    over elements of the sequence.
    """
    def __init__(self,
                 vocabulary_size: int,
                 embedding_size: int,
                 pooling_type: Optional[str] = None):
        """

        :param vocabulary_size: the size of the elements vocabulary.
        :param embedding_size: the embedding size.
        :param pooling_type: the type of pooling that will be performed. (max, sum or mean)
        """
        super().__init__()
        self.elements_embedding = SequenceElementsEmbeddingLayer(vocabulary_size, embedding_size, pooling_type)

    def forward(self, input_sequence: InputSequence) -> EmbeddedElementsSequence:
        embedded_sequence = self.elements_embedding(input_sequence.sequence)

        return EmbeddedElementsSequence(embedded_sequence, input_sequence)
