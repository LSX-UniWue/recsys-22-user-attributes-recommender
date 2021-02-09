from typing import Dict, Optional, List

import torch

from metrics.container.metrics_container import MetricsContainer
from metrics.ranking.ranking_metric import RankingMetric


class RankingMetricsContainer(MetricsContainer):

    def __init__(self, metrics: List[RankingMetric]):
        """
        Creates a container with the given metrics.

        :param metrics: a list of metrics.
        """
        super().__init__()
        self.metrics = torch.nn.ModuleList(metrics)

    def update(self,
               input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Updates all metrics stored in :code self.metrics using the provided inputs and predictions.

        :param input_seq: the batch with input sequences.
        :param targets: the expected targets.
        :param predictions: the predictions made by the model.
        :param mask: ???

        :return: a dictionary with the step values for all metrics managed by this module.
        """
        results = {}
        for metric in self.metrics:
            step_value = metric(predictions, targets, mask)
            results[metric.name()] = step_value  #TODO add name to ranking metrics

    def compute(self) -> Dict[str, torch.Tensor]:
       """
       Computes the final results for all metrics managed by this module.
       :return: the final metric values.
       """
       return {metric.name(): metric.compute() for metric in self.metrics}
