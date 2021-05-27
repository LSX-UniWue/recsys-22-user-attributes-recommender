from abc import ABC
from typing import Union, Tuple, Optional, List, Dict

import torch
from torch import nn

from asme.models.layers.layers import SequenceElementsRepresentationLayer, SequenceRepresentationLayer, \
    SequenceRepresentationModifierLayer, ProjectionLayer


class IdentitySequenceRepresentationModifierLayer(SequenceRepresentationModifierLayer):

    """ a SequenceRepresentationModifierLayer that does nothing with the sequence representation """

    def forward(self,
                sequence_representation: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                **kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sequence_representation


class SequenceRecommenderModel(ABC, nn.Module):
    """
    Configurable sequence recommender model.
    """
    def __init__(self,
                 sequence_embedding_layer: SequenceElementsRepresentationLayer,
                 sequence_representation_layer: SequenceRepresentationLayer,
                 sequence_representation_modifier_layer: SequenceRepresentationModifierLayer,
                 projection_layer: ProjectionLayer):
        """

        :param sequence_embedding_layer: a layer that computes an embedding for the items in a sequence.`
        :param sequence_representation_layer:  a layer that computes a representation for a sequence.
        :param sequence_representation_modifier_layer: an optional layer that transforms the sequence representation.
        :param projection_layer: a layer that computes scores for items based on the sequence representation.
        """
        super().__init__()

        self._sequence_embedding_layer = sequence_embedding_layer
        self._sequence_representation_layer = sequence_representation_layer
        self._sequence_representation_modifier_layer = sequence_representation_modifier_layer
        self._projection_layer = projection_layer

    def forward(self,
                sequence: torch.Tensor,
                padding_mask: Optional[torch.Tensor],
                **kwargs
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param sequence: a sequence tensor. :math:`(N, S)` or :math:`(N, S, BS)`
        :param padding_mask: a mask that contains positions of padding tokens. :math:`(N, S)`
        :param kwargs:
        :return:
        """
        # 1. embed the information
        embedded_sequence = self._sequence_embedding_layer(sequence, padding_mask, **kwargs)

        # 2. get the sequence representation
        sequence_representation = self._sequence_representation_layer(embedded_sequence, padding_mask, **kwargs)

        # 3. maybe modify the representation
        sequence_representation = self._sequence_representation_modifier_layer(sequence_representation, padding_mask, **kwargs)

        return self._projection_layer(sequence_representation, padding_mask, **kwargs)

    def required_metadata_keys(self) -> List[str]:
        """
        Returns a list of keys that reference metadata in a batch that is required to apply the model.

        :return: a list with keys that reference required metadata in a batch. Default: []
        """
        return []

    def optional_metadata_keys(self) -> List[str]:
        """
        Returns a list of keys that reference metadata in a batch that is optional to apply the model.

        :return: a list with keys that reference optional metadata in a batch. Default: []
        """
        return []
