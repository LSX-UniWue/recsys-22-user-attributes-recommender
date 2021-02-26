from typing import List

from numpy.random._generator import default_rng
from torch.utils.data import Dataset, IterableDataset

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, SAMPLE_IDS
from data.datasets.index import SessionPositionIndex
from data.datasets.processors.processor import Processor
from data.datasets.session import ItemSessionDataset, PlainSessionDataset
from data.mp import MultiProcessSupport


# Todo find better name
class NextItemDataset(Dataset, MultiProcessSupport):

    def __init__(self,
                 dataset: PlainSessionDataset,
                 index: SessionPositionIndex,
                 processors: List[Processor] = None,
                 add_target: bool = True,
                 include_target_pos: bool = False
                 ):
        super().__init__()
        self._dataset = dataset
        self._index = index
        if processors is None:
            processors = []
        self._processors = processors
        self._add_target = add_target
        self._include_target_pos = include_target_pos

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        session_idx, target_pos = self._index[idx]
        parsed_session = self._dataset[session_idx]
        parsed_session[SAMPLE_IDS] = session_idx
        parsed_session['pos'] = target_pos

        session = parsed_session[ITEM_SEQ_ENTRY_NAME]
        truncated_session = session[:target_pos] if not self._include_target_pos else session[:target_pos + 1]
        parsed_session[ITEM_SEQ_ENTRY_NAME] = truncated_session
        if self._add_target:
            parsed_session[TARGET_ENTRY_NAME] = session[target_pos]

        for processor in self._processors:
            parsed_session = processor.process(parsed_session)

        return parsed_session

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        # nothing to do here
        pass


class NextItemIterableDataset(IterableDataset):
    def __init__(self, dataset: ItemSessionDataset, index: SessionPositionIndex, seed: int = None):
        self._dataset = dataset
        self._index = index
        self._seed = seed

        self._start = 0
        self._stop = len(self._index)

    def __iter__(self):
        rng = default_rng(self._seed)

        while True:
            sample_idx = rng.integers(low=self._start, high=self._stop)
            session_idx, target_idx = self._index[sample_idx]

            session = self._dataset[session_idx][ITEM_SEQ_ENTRY_NAME]

            yield {
                ITEM_SEQ_ENTRY_NAME: session[:target_idx],
                TARGET_ENTRY_NAME: session[target_idx],
                SAMPLE_IDS: sample_idx
            }

    def __len__(self):
        return len(self._index)
