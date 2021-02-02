from abc import abstractmethod
from typing import Dict, Optional, List

import torch

from torch import nn

# (AD) tagged as a pytorch module so PL recognizes it an performs things like moving state between devices
from metrics.container.metrics_sampler import MetricsSampler
from metrics.metric import RankingMetric


class MetricsContainer(nn.Module):
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


class NoopMetricsContainer(MetricsContainer):

    def update(self, input_seq: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        return {}

    def compute(self) -> Dict[str, torch.Tensor]:
        return {}


class RankingMetricsContainer(MetricsContainer):
    """
    A module that can be used as a container for a collection of metrics.
    """
    def __init__(self,
                 metrics: List[RankingMetric],
                 sampler: MetricsSampler):
        """
        Construtor.

        :param metrics: a list of sampling metrics.
        :param sampler: the sampler that should be used to draw the negative samples.
        """
        super().__init__()
        self.metrics = torch.nn.ModuleList(metrics)  # making sure that all metrics are recognized as modules
        self.sampler = sampler

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

        samples = self.sampler.sample(input_seq, targets, predictions, mask)

        results = {}

        for metric in self.metrics:
            step_value = metric(samples.sampled_predictions, samples.positive_item_mask)
            results[f'{metric.name()}{self.sampler.suffix_metric_name()}'] = step_value

        return results

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Computes the final results for all metrics managed by this module.
        :return: the final metric values.
        """
        return {metric.name(): metric.compute() for metric in self.metrics}


class AggregateMetricsContainer(MetricsContainer):
    """
    Manages a list of containers each managing different metrics.
    Forwards calls to all containers and aggregates results into a single dictionary.
    """
    def __init__(self, containers: List[MetricsContainer]):
        super().__init__()
        self.containers = torch.nn.ModuleList(containers)

    def update(self, input_seq: torch.Tensor,
               targets: torch.Tensor,
               predictions: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        results = {}

        for container in self.containers:
            container_results = container.update(input_seq, targets, predictions, mask)
            if container_results is not None:
                results.update(container_results)

        return results

    def compute(self) -> Dict[str, torch.Tensor]:
        results = {}

        for container in self.containers:
            container_results = container.compute()
            if container_results is not None:
                results.update(container_results)

        return results
