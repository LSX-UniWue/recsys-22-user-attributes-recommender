from typing import List

import torch

from metrics.ranking_metrics import RecallAtMetric, MRRAtMetric


def build_metrics(ks: List[int]):
    metrics = {}

    for k in ks:
        metrics[f"recall_at_{k}"] = RecallAtMetric(k)
        metrics[f"mrr_at_{k}"] = MRRAtMetric(k)

    return torch.nn.ModuleDict(modules=metrics)
