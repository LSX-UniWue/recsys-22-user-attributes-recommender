import abc
from abc import ABC
from typing import Union, Tuple, Optional, List, Dict

import torch
from torch import nn


class SequenceElementsRepresentationLayer(ABC, nn.Module):
    """
    Base class for modules that embed the elements of a sequence.
    """
    @abc.abstractmethod
    def forward(self,
                sequence: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                **kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        :param sequence: a sequence tensor with item ids. :math:`(N, S)` or :math:`(N, S, BS)`
        :param padding_mask: a mask that contains positions of padding tokens. :math:`(N, S)`
        :param kwargs: attributes that can be used to contextualize the sequence

        :return: a sequence with embedded elements. :math:`(N, S, H)`
        """
        pass


class SequenceRepresentationLayer(ABC, nn.Module):
    def forward(self,
                sequence: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                **kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param sequence: an embedded sequence tensor. :math:`(N, S, H)`
        :param padding_mask: a mask that contains positions of padding tokens. :math:`(N, S)`
        :param kwargs: attributes that can be used to contextualize the sequence representation.

        :return: a sequence representation. :math:`(N, S, R)`
        """
        pass


class SequenceRepresentationModifierLayer(ABC, nn.Module):
    def forward(self,
                sequence: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                **kwargs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param sequence: a sequence tensor. :math:`(N, S, R)`
        :param padding_mask: a mask that contains positions of padding tokens. :math:`(N, S)`
        :param kwargs: attributes that can be used to contextualize the sequence

        :return: a sequence with embedded elements. :math:`(N, S, T)`
        """
        pass


class ProjectionLayer(ABC, nn.Module):
    def forward(self,
                sequence_representation: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                positive_samples: Optional[torch.Tensor] = None,
                negative_samples: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        :param sequence_representation: a sequence tensor. :math:`(N, S, T)`
        :param padding_mask: a mask that contains positions of padding tokens. :math:`(N, S)`
        :param positive_samples: a tensor with positive sample item ids to score. :math:`(N, P)`
        :param negative_samples: a tensor with negative sample item ids o score. :math:`(N, NI)`

        :return: a score for each (provided) item.
        if no positive_samples and negative samples are provided a tensor of :math:`(N, I)` is returned
        if positive samples are provided :math:`(N, PI)` or
        if positive and negative samples are provided a tuple of two tensors of shape :math:`(N, PI)`, :math:`(N, NI)`
        """
        pass


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
