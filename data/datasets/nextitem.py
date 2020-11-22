from typing import List

from numpy.random._generator import default_rng
from torch.utils.data import Dataset, IterableDataset

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from data.datasets.index import SessionPositionIndex
from data.datasets.prepare import Processor
from data.datasets.session import ItemSessionDataset, PlainSessionDataset
from data.mp import MultiProcessSupport


class NextItemDataset(Dataset, MultiProcessSupport):

    def __init__(self,
                 dataset: PlainSessionDataset,
                 index: SessionPositionIndex,
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
        session_idx, target_pos = self._index[idx]
        parsed_session = self._dataset[session_idx]
        session = parsed_session[ITEM_SEQ_ENTRY_NAME]
        truncated_session = session[:target_pos]
        parsed_session[ITEM_SEQ_ENTRY_NAME] = truncated_session
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
                TARGET_ENTRY_NAME: session[target_idx]
            }

    def __len__(self):
        return len(self._index)
