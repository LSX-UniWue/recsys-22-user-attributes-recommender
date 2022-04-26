from typing import List

from asme.core.evaluation.evaluation import BatchEvaluator
from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.common.dependencies_factory import DependenciesFactory
from asme.core.init.factories.evaluation.evaluators import EvaluatorsFactory
from asme.core.init.factories.evaluation.item_filter import FilterPredictionItemsFactory

from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class EvaluationFactory(ObjectFactory):
    """
    Builds all meta information within the `evaluation` section.
    """
    KEY = "evaluation"

    def __init__(self):
        super().__init__()

        self.filter_factory = DependenciesFactory([FilterPredictionItemsFactory()])
        self.evaluators_factory = DependenciesFactory([EvaluatorsFactory()])

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:

        can_build_result = self.filter_factory.can_build(config, context)
        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        can_build_result = self.evaluators_factory.can_build(config, context)
        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        return can_build_result

    def build(self,
              config: Config,
              context: Context
              ) -> List[BatchEvaluator]:

        num_predictions = config.get_or_default("number_predictions", 20)

        #TODO Add default evaluators?
        #config.get("evaluators").append({"type": "sid"})
        #config.get("evaluators").append({"type": "recommendation"})

        for evaluator in config.get("evaluators"):
           evaluator["number_predictions"] = num_predictions

        filter_items = self.filter_factory.build(config, context)
        context.set(self.config_path(), filter_items)

        evaluation = self.evaluators_factory.build(config, context)
        evaluation.update(filter_items)

        context.set(self.config_path(), evaluation, overwrite=True)

        return evaluation

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
