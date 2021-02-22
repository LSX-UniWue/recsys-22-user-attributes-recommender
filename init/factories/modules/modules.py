from typing import Any, List

from init.config import Config
from init.context import Context
from init.factories.metrics.metrics_container import MetricsContainerFactory
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ModuleFactory(ObjectFactory):

    KEY = "module"

    def __init__(self):
        super().__init__()
        self.metrics_container_factory = MetricsContainerFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        metrics_config = config.get_config(self.metrics_container_factory.config_path())

        can_build_metrics = self.metrics_container_factory.can_build(metrics_config, context)
        if can_build_metrics.type != CanBuildResultType.CAN_BUILD:
            return can_build_metrics

        # FIXME: implement

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Any:

        metrics = self.metrics_container_factory.build(config.get_config(self.metrics_container_factory.config_path()), context)

        return self.elements_factory.build(config, context)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
