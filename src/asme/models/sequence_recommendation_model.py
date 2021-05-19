from typing import Union, Tuple, Optional

import torch
from torch import nn


class SequenceRecommenderModel(nn.Module):
    """
    base class to implement a sequence recommendation model
    """
    def forward(self,
                sequence: torch.Tensor,
                padding_mask: Optional[torch.Tensor],
                **kwargs
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass
