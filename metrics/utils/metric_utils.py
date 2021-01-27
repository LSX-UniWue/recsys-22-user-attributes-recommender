import pytorch_lightning as pl

from typing import List, Dict, Union, Callable

import torch

from metrics.rank.ndcg import NormalizedDiscountedCumulativeGain
from metrics.rank.dcg import DiscountedCumulativeGain
from metrics.sample.recall_at import RecallAtNegativeSamples
from metrics.rank.mrr_at import MRRAtMetric
from metrics.rank.recall_at import RecallAtMetric
from metrics.rank.precision_at import PrecisionAtMetric
from metrics.rank.f1_at import F1AtMetric


def _build_metric(metric_id: str
                  , k: int) -> pl.metrics.Metric:
    return {
        'recall_sampled': RecallAtNegativeSamples(k),
        'recall': RecallAtMetric(k),
        'precision': PrecisionAtMetric(k),
        'f1': F1AtMetric(k),
        'mrr': MRRAtMetric(k),
        'dcg': DiscountedCumulativeGain(k),
        'ndcg': NormalizedDiscountedCumulativeGain(k),
    }[metric_id]


def _build_sampled_metric(metric_id: str
                          , k: int) -> pl.metrics.Metric:
    return {
        'recall': RecallAtNegativeSamples(k),
    }[metric_id]


def _build_metrics_module_dict(metric_dict: Dict[str, Union[int, List[int]]],
                               build_metric: Callable[[str, int], pl.metrics.Metric]
                               ) -> torch.nn.ModuleDict:
    metrics = {}

    if metric_dict is None:
        return metrics

    for metric_id, ks in metric_dict.items():
        # ks can be a single k => first convert it to a list
        if not isinstance(ks, list):
            ks = [ks]
        for k in ks:
            metrics[f"{metric_id}_at_{k}"] = build_metric(metric_id, k)

    return torch.nn.ModuleDict(modules=metrics)


def build_metrics(metric_dict: Dict[str, Union[int, List[int]]]
                  ) -> torch.nn.ModuleDict:
    return _build_metrics_module_dict(metric_dict, _build_metric)


def build_sampled_metrics(metric_dict: Dict[str, Union[int, List[int]]]
                          ) -> torch.nn.ModuleDict:
    return _build_metrics_module_dict(metric_dict, _build_sampled_metric)
