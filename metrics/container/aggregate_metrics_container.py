from typing import Dict, Optional, List

import torch

from metrics.container.metrics_container import MetricsContainer


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
