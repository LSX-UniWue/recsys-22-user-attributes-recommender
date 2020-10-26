import pytorch_lightning as pl

from typing import List, Dict, Union

import torch

from metrics.ranking_metrics import PrecisionAtMetric, RecallAtMetric, MRRAtMetric


def _build_metric(metric_id: str
                  , k: int) -> pl.metrics.Metric:
    return {
        'recall': RecallAtMetric(k),
        'precision': PrecisionAtMetric(k),
        'mrr': MRRAtMetric(k)
    }[metric_id]


def build_metrics(metric_dict: Dict[str, Union[int, List[int]]]
                  ) -> torch.nn.ModuleDict:
    metrics = {}

    for metric_id, ks in metric_dict.items():
        # ks can be a single k => first convert it to a list
        if not isinstance(ks, list):
            ks = [ks]
        for k in ks:
            metrics[f"{metric_id}_at_{k}"] = _build_metric(metric_id, k)

    return torch.nn.ModuleDict(modules=metrics)
