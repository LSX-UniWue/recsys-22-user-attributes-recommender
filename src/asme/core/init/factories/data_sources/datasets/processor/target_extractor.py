import distutils
from typing import List
from asme.core.init.factories.features.features_factory import FeaturesFactory

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.data.datasets.processors.target_extractor import TargetExtractorProcessor


class TargetExtractorProcessorFactory(ObjectFactory):
    """
    Factory for the PositionTokenProcessor
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> TargetExtractorProcessor:

        features = context.get([FeaturesFactory.KEY])

        if config.has_path(["parallel"]):
            parallel_flag = config.get(["parallel"])
            parallel = parallel_flag if isinstance(parallel_flag, bool) else distutils.util.strtobool(config.get(["parallel"]))
        else:
            parallel = False

        if config.has_path(["first_target"]):
            first_target_flag = config.get(["first_target"])
            first_target = first_target_flag if isinstance(first_target_flag, bool) else distutils.util.strtobool(config.get(["first_target"]))
        else:
            first_target = False

        return TargetExtractorProcessor(features, parallel, first_target)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'target_extractor'
