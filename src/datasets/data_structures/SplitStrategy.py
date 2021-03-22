import abc
from datasets.data_structures.TrainValidationTestSplitsIndices import TrainValidationTestSplitIndices


class SplitStrategy(abc.ABC):
    @abc.abstractmethod
    def split(self, dataset_metadata) -> TrainValidationTestSplitIndices:
        pass
