from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.writer.prediction.evaluator_prediction_writer import CSVMultiLineWriter


class CSVMultiLineWriterFactory(ObjectFactory):
    """
    Factory for the CSVMultiLineWriter
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> CSVMultiLineWriter:

        evaluators = context.get("evaluation")["evaluators"]
        output
        return CSVMultiLineWriter(evaluators)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return ["writer"]

    def config_key(self) -> str:
        return 'evaluation'
