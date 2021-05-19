from abc import ABC
from typing import Union, Tuple, Optional, List

import torch
from torch import nn


class SequenceRecommenderModel(ABC, nn.Module):
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
