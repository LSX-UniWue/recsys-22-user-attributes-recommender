import abc
from datasets.data_structures.train_validation_test_splits_indices import TrainValidationTestSplitIndices


class SplitStrategy(abc.ABC):
    @abc.abstractmethod
    def split(self, dataset_metadata) -> TrainValidationTestSplitIndices:
        pass
