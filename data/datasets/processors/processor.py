from abc import abstractmethod
from typing import Dict, Any, List


class Processor:

    """
    Processors can be used to augment the raw input data. Examples include: masking tokens or augmenting the input data.
    """

    @abstractmethod
    def process(self,
                input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes the input and processes/enhances it.

        :param input: input data at this step of the pipeline.
        :return: a dictionary with the processed version of the input data.
        """
        pass


class DelegatingProcessor(Processor):

    def __init__(self,
                 processors: List[Processor]):
        super().__init__()

        self.processors = processors

    def process(self,
                input: Dict[str, Any]) -> Dict[str, Any]:
        for processor in self.processors:
            input = processor.process(input)

        return input
