from typing import List

from torch.utils.data import Dataset

from data.datasets import SAMPLE_IDS, ITEM_SEQ_ENTRY_NAME
from data.datasets.index import SequencePositionIndex
from data.datasets.processors.processor import Processor
from data.datasets.sequence import PlainSequenceDataset
from data.mp import MultiProcessSupport


class SequencePositionDataset(Dataset, MultiProcessSupport):

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

    def __getitem__(self, idx):
        sequence_idx, position = self._index[idx]
        parsed_session = self._dataset[sequence_idx]
        parsed_session[SAMPLE_IDS] = sequence_idx
        parsed_session['pos'] = position

        parsed_session[ITEM_SEQ_ENTRY_NAME] = parsed_session[ITEM_SEQ_ENTRY_NAME][:position + 1]

        for processor in self._processors:
            parsed_session = processor.process(parsed_session)

        return parsed_session

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        # nothing to do here
        pass
