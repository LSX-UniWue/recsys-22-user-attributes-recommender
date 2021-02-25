from typing import Dict, Any, Union, List, Callable

import pytorch_lightning as pl
from dependency_injector import providers

from metrics.container.metrics_container import AggregateMetricsContainer, RankingMetricsContainer, \
    NoopMetricsContainer, MetricsContainer
from metrics.container.metrics_sampler import NegativeMetricsSampler, FixedItemsSampler, AllItemsSampler
from metrics.dcg import DiscountedCumulativeGainMetric
from metrics.ndcg import NormalizedDiscountedCumulativeGainMetric
from metrics.precision import PrecisionMetric
from metrics.recall import RecallMetric
from metrics.f1 import F1Metric
from metrics.mrr import MRRMetric


def build_aggregate_metrics_container(config: providers.Configuration) -> providers.Factory:
    """
    Builds an AggregateMetricsContainer with all metrics defined in the module section of the configuration file.
    :param config: the module section of the configuration file.

    :return: a factory that builds an AggregateMetricsContainer
    """
    return providers.Factory(
        AggregateMetricsContainer,
        build_all_metrics_containers(config)
    )


def build_all_metrics_containers(config: providers.Configuration) -> providers.List:
    return providers.List(
        build_ranking_metrics_provider(config),
        build_sampling_metrics_provider(config),
        build_fixed_sampling_metrics_provider(config)
    )


def build_ranking_metrics_provider(config: providers.Configuration):
    return providers.Factory(_build_ranking_metrics_provider, config)


def _build_ranking_metrics_provider(config: Dict[str, Any]) -> MetricsContainer:
    if "metrics" not in config:
        return NoopMetricsContainer()

    return RankingMetricsContainer(_build_metrics(config["metrics"]), AllItemsSampler())


def _build_ranking_metrics(metric_dict: Dict[str, Union[int, List[int]]]
                  ) -> List[pl.metrics.Metric]:
    return _build_metrics_list(metric_dict, _build_ranking_metric)


def build_sampling_metrics_provider(config: providers.Configuration):
    return providers.Factory(_build_sampling_metrics_container, config)


def _build_sampling_metrics_container(config: Dict[str, Any]) -> MetricsContainer:
    if "sampled_metrics" not in config:
        return NoopMetricsContainer() #FIXME I don't want to waste time doing this properly since we will remove the DI framework anyways.

    negative_sampler = NegativeMetricsSampler(
        load_weights(config["sampled_metrics"]["sample_probability_file"]),
        int(config["sampled_metrics"]["num_negative_samples"])
    )
    return RankingMetricsContainer(_build_metrics(config["sampled_metrics"]["metrics"]), negative_sampler)


def build_fixed_sampling_metrics_provider(config: providers.Configuration):
    return providers.Factory(_build_fixed_sampling_metrics_container, config)


def _build_fixed_sampling_metrics_container(config: Dict[str, Any]) -> MetricsContainer:
    if "fixed_subset_metrics" not in config:
        return NoopMetricsContainer()

    fixed_config = config["fixed_subset_metrics"]
    sampler = FixedItemsSampler(load_items(fixed_config["item_file"]))

    return RankingMetricsContainer(_build_metrics(fixed_config['metrics']), sampler)


def _build_metrics(metric_dict: Dict[str, Union[int, List[int]]]
                   ) -> List[pl.metrics.Metric]:
    return _build_metrics_list(metric_dict, _build_ranking_metric)


def _build_metrics_list(metric_dict: Dict[str, Union[int, List[int]]],
                        build_metric: Callable[[str, int], pl.metrics.Metric]) -> List[pl.metrics.Metric]:
    metrics = []
    if metric_dict is None:
        return metrics

    for metric_id, ks in metric_dict.items():
        # ks can be a single k => first convert it to a list
        if not isinstance(ks, list):
            ks = [ks]
        for k in ks:
            metrics.append(build_metric(metric_id, k))

    return metrics


def _build_ranking_metric(metric_id: str, k: int) -> pl.metrics.Metric:
    return {
        'recall': RecallMetric(k),
        'precision': PrecisionMetric(k),
        'f1': F1Metric(k),
        'mrr': MRRMetric(k),
        'dcg': DiscountedCumulativeGainMetric(k),
        'ndcg': NormalizedDiscountedCumulativeGainMetric(k),
    }[metric_id]


def load_weights(path: str) -> List[float]:
    with open(path) as prob_file:
        return [float(line) for line in prob_file.readlines()]


def load_items(path: str) -> List[int]:
    with open(path) as item_file:
        return [int(line) for line in item_file.readlines()]
