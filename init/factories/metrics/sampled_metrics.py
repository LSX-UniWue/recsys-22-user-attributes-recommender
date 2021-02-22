from typing import List

from init.config import Config
from init.context import Context
from init.factories.util import require_config_keys
from init.object_factory import ObjectFactory, CanBuildResult
from metrics.container.sampling_metrics_container import SamplingMetricsContainer


class SampledMetricsFactory(ObjectFactory):
    KEY = 'sampled'

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return require_config_keys(config, ['metrics', 'num_negative_samples', 'sample_probability_file'])

    def build(self, config: Config, context: Context) -> SamplingMetricsContainer:

        # FIXME: implement metrics factory
        metrics = []
        return SamplingMetricsContainer(metrics)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
