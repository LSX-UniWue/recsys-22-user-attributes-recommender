from typing import List, Union, Any, Dict

from asme.core.evaluation.evaluation import BatchEvaluator
from asme.core.evaluation.registry import REGISTERED_EVALUATORS
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.core.init.factories.common.config_based_factory import ListFactory

from asme.core.init.object_factory import ObjectFactory, CanBuildResult
from asme.core.writer.prediction.registry import REGISTERED_WRITERS


class WritersFactory(ObjectFactory):
    """
    Builds the writer
    """
    KEY = "writer"

    def __init__(self):
        super().__init__()

        self.writer_factory = ConditionalFactory(
                'type',
                REGISTERED_WRITERS
            )


    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return self.writer_factory.can_build(config, context)

    def build(self,
              config: Config,
              context: Context
              ) -> List[BatchEvaluator]:
        return self.writer_factory.build(config, context)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return ["writer"]

    def config_key(self) -> str:
        return self.KEY
