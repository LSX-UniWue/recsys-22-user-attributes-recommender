import pytorch_lightning as pl

from typing import List

import torch

from metrics.ranking_metrics import PrecisionAtMetric, RecallAtMetric, MRRAtMetric


def _build_metric(metric_id: str
                  , k: int) -> pl.metrics.Metric:
    return {
        'recall': RecallAtMetric(k),
        'precision': PrecisionAtMetric(k),
        'mrr': MRRAtMetric(k)
    }[metric_id]


def build_metrics(metric_ids: List[str],
                  ks: List[int]
                  ) -> torch.nn.ModuleDict:
    metrics = {}

    for k in ks:
        for metric_id in metric_ids:
            metrics[f"{metric_id}_at_{k}"] = _build_metric(metric_id, k)

    return torch.nn.ModuleDict(modules=metrics)
