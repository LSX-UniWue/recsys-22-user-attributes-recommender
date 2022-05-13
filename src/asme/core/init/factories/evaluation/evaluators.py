from typing import List, Union, Any, Dict

from asme.core.evaluation.evaluation import BatchEvaluator
from asme.core.evaluation.registry import REGISTERED_EVALUATORS
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.core.init.factories.common.config_based_factory import ListFactory

from asme.core.init.object_factory import ObjectFactory, CanBuildResult


class EvaluatorsFactory(ObjectFactory):
    """
    Builds all meta information within the `features` section.
    """
    KEY = "evaluators"

    def __init__(self):
        super().__init__()

        self.evaluator_factories = ListFactory(
            ConditionalFactory(
                'type',
                REGISTERED_EVALUATORS
            )
        )


    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return self.evaluator_factories.can_build(config, context)

    def build(self,
              config: Config,
              context: Context
              ) -> List[BatchEvaluator]:
        return self.evaluator_factories.build(config, context)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return ['evaluators']

    def config_key(self) -> str:
        return self.KEY
