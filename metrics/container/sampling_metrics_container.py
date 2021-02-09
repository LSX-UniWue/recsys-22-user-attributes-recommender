from typing import List, Optional, Dict

import torch

from metrics.container.metrics_sampler import MetricsSampler
from metrics.container.metrics_container import MetricsContainer
from metrics.sampling.sampling_metric import SamplingMetric


class SamplingMetricsContainer(MetricsContainer):
    """
    A module that can be used as a container for a collection of sampling metrics.
    """
    def __init__(self,
                 metrics: List[SamplingMetric],
                 negative_sampler: MetricsSampler):
        """
        Construtor.

        :param metrics: a list of sampling metrics.
        :param negative_sampler: the sampler that should be used to draw the negative samples.
        """
        super().__init__()
        self.metrics = torch.nn.ModuleList(metrics)  # making sure that all metrics are recognized as modules
        self.negative_sample = negative_sampler

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
        negative_sample = self.negative_sample.sample(input_seq, targets, predictions, mask)

        results = {}

        for metric in self.metrics:
            step_value = metric(negative_sample.sampled_predictions, negative_sample.positive_item_mask)
            results[metric.name()] = step_value

        return results

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes the final results for all metrics managed by this module.
        :return: the final metric values.
        """
        return {metric.name(): metric.compute() for metric in self.metrics}
