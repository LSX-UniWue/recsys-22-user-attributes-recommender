from abc import abstractmethod
from typing import Dict, Optional

import torch


# (AD) tagged as a pytorch module so PL recognizes it an performs things like moving state between devices
class MetricsContainer(torch.nn.Module):
    """
    Base class for containers that manage a collection of metrics.
    """
    @abstractmethod
    def update(self,
               input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Updates all metrics managed by this container using the provided inputs and predictions.

        :param input_seq: the batch with input sequences.
        :param targets: the expected targets.
        :param predictions: the predictions made by the model.
        :param mask: ???

        :return: a dictionary with the step values for all metrics managed by this module.
        """
        pass

    @abstractmethod
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes the final results for all metrics managed by this container.
        :return: the final metric values.
        """
        pass
