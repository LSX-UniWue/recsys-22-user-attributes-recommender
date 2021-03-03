from typing import List

from torch.utils.data import Dataset

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, SAMPLE_IDS
from data.datasets.index import SequencePositionIndex
from data.datasets.processors.processor import Processor
from data.datasets.sequence import PlainSequenceDataset
from data.mp import MultiProcessSupport


# Todo find better name
class SequencePositionDataset(Dataset, MultiProcessSupport):

    """
    A dataset that uses a sequence position index to load all session upon the specified position in the index

    if add_target is set to False, the complete sequence till the position is returned else the sequence excluding the
    position is returned and the
    """

    def __init__(self,
                 dataset: PlainSequenceDataset,
                 index: SequencePositionIndex,
                 processors: List[Processor] = None,
                 add_target: bool = True
                 ):
        super().__init__()
        self._dataset = dataset
        self._index = index
        if processors is None:
            processors = []
        self._processors = processors
        self._add_target = add_target

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        session_idx, target_pos = self._index[idx]
        parsed_session = self._dataset[session_idx]
        parsed_session[SAMPLE_IDS] = session_idx
        parsed_session['pos'] = target_pos

        session = parsed_session[ITEM_SEQ_ENTRY_NAME]
        truncated_session = session[:target_pos] if self._add_target else session[:target_pos + 1]
        parsed_session[ITEM_SEQ_ENTRY_NAME] = truncated_session
        if self._add_target:
            parsed_session[TARGET_ENTRY_NAME] = session[target_pos]

        for processor in self._processors:
            parsed_session = processor.process(parsed_session)

        return parsed_session

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        # nothing to do here
        pass
