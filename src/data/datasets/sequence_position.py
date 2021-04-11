import random
from typing import List, Any, Dict

from torch.utils.data import Dataset

from data.datasets import SAMPLE_IDS, ITEM_SEQ_ENTRY_NAME
from data.datasets.index import SequencePositionIndex
from data.datasets.processors.processor import Processor
from data.datasets.sequence import PlainSequenceDataset
from data.examle_logging import ExampleLogger, Example
from data.multi_processing import MultiProcessSupport


class SequencePositionDataset(Dataset, MultiProcessSupport, ExampleLogger):

    """
    A dataset that uses a sequence position index to load the session till the specified position in the index

    e.g. if the csv contains the sequence [9, 5, 6, 7] and the position 2 in the position index the sequence
    [9, 5, 6] is returned
    """

    def __init__(self,
                 dataset: PlainSequenceDataset,
                 index: SequencePositionIndex,
                 processors: List[Processor] = None
                 ):
        super().__init__()
        self._dataset = dataset
        self._index = index
        if processors is None:
            processors = []
        self._processors = processors

    def __len__(self):
        return len(self._index)

    def _get_example(self, idx: int) -> Example:
        sequence_data = self._get_raw_sequence(idx)
        processed_data = self._process_sequence(sequence_data.copy())

        return Example(sequence_data, processed_data)

    def _get_raw_sequence(self, idx: int) -> Dict[str, Any]:
        sequence_idx, position = self._index[idx]
        parsed_session = self._dataset[sequence_idx]
        parsed_session[SAMPLE_IDS] = sequence_idx
        parsed_session['pos'] = position

        parsed_session[ITEM_SEQ_ENTRY_NAME] = parsed_session[ITEM_SEQ_ENTRY_NAME][:position + 1]
        return parsed_session

    def _process_sequence(self,
                          parsed_sequence: Dict[str, Any]
                          ) -> Dict[str, Any]:
        for processor in self._processors:
            parsed_sequence = processor.process(parsed_sequence)

        return parsed_sequence

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self._get_example(idx)
        return example.processed_data

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        # nothing to do here
        pass

    def get_data_examples(self, num_examples: int = 1) -> List[Example]:
        return [
            self._get_example(example_id) for example_id in random.sample(range(0, len(self)), num_examples)
        ]
