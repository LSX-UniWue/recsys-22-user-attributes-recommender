import abc

from datasets.data_structures.dataset_metadata import DatasetMetadata
from datasets.data_structures.train_validation_test_splits_indices import TrainValidationTestSplitIndices


class SplitStrategy:

    @abc.abstractmethod
    def split(self, dataset_metadata: DatasetMetadata) -> TrainValidationTestSplitIndices:
        pass
