from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.writer.prediction.evaluator_prediction_writer import CSVMultiLineWriter, CSVSingleLineWriter


class CSVSingleLineWriterFactory(ObjectFactory):
    """
    Factory for the CSVSingleeLineWriter
    """

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> CSVSingleLineWriter:

        evaluators = context.get("evaluation")["evaluators"]
        return CSVSingleLineWriter(evaluators)

    def is_required(self, context: Context) -> bool:
        return False

    def config_path(self) -> List[str]:
        return ["writer"]

    def config_key(self) -> str:
        return 'evaluation'
