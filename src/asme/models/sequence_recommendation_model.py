from abc import ABC
from typing import Union, Tuple, Optional, List

import torch
from torch import nn

from asme.models.layers.layers import ItemEmbedding


class SequenceRepresentationLayer(ABC, nn.Module):
    pass


class ProjectionLayer(ABC, nn.Module):
    pass


class SequenceRepresentationModifierLayer(ABC, nn.Module):
    pass


class SequenceRecommenderModel(ABC, nn.Module):
    def __init__(self,
                 item_embedding_layer: ItemEmbedding,  # rename ItemEmbeddingLayer
                 sequence_representation_layer: SequenceRepresentationLayer,
                 sequence_representation_modifier_layer: SequenceRepresentationModifierLayer,
                 projection_layer: ProjectionLayer):
        super().__init__()
        # TODO

    """
    base class to implement a sequence recommendation model
    """
    def forward(self,
                sequence: torch.Tensor,
                padding_mask: Optional[torch.Tensor],
                **kwargs
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass

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
