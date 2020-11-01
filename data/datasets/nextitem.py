import io
import math
from pathlib import Path
import sys
from typing import Tuple

from numpy.random._generator import default_rng
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

from data.datasets import ITEM_SEQ_ENTRY_NAME, INT_BYTE_SIZE, TARGET_ENTRY_NAME
from data.datasets.session import ItemSessionDataset
from data.mp import MultiProcessDataLoaderSupport


class NextItemIndexBuilder:

    def __init__(self, min_session_length: int = 2):
        self._min_session_length = min_session_length

    def build(self, dataset: ItemSessionDataset, index_path: Path):
        if not index_path.exists():
            index_path.parent.mkdir(parents=True, exist_ok=True)

        current_idx = 0
        with index_path.open("wb") as index_file:
            for session_idx in tqdm(range(len(dataset)), desc="Creating Index."):
                sessions = dataset[session_idx][ITEM_SEQ_ENTRY_NAME]
                # remove sessions with
                if len(sessions) > self._min_session_length:
                    for target_pos in range(1, len(sessions)):
                        self._write_entry(index_file, session_idx, target_pos)
                        current_idx += 1
            # write length at the end
            index_file.write(current_idx.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))
            # write minimum length
            index_file.write(self._min_session_length.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))

    @staticmethod
    def _write_entry(index_file, session_idx: int, target_pos: int):
        index_file.write(session_idx.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))
        index_file.write(target_pos.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))


class NextItemIndex(MultiProcessDataLoaderSupport):
    def __init__(self, index_path: Path):
        if not index_path.exists():
            raise Exception(f"could not find file with index at: {index_path}")
        self._index_path = index_path

        self._init()

    def _init(self):
        self._index_file_handle = self._index_path.open("rb")
        self._min_session_length = self._read_min_session_length()
        self._length = self._read_length()

    def _read_length(self):
        self._index_file_handle.seek(-2 * INT_BYTE_SIZE, io.SEEK_END)
        return int.from_bytes(self._index_file_handle.read(INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

    def _read_min_session_length(self):
        self._index_file_handle.seek(-INT_BYTE_SIZE, io.SEEK_END)
        return int.from_bytes(self._index_file_handle.read(INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self._index_file_handle.seek(idx * INT_BYTE_SIZE * 2)
        session_idx = int.from_bytes(self._index_file_handle.read(INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)
        target_pos = int.from_bytes(self._index_file_handle.read(INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

        return session_idx, target_pos

    def _mp_init(self, id: int, num_worker: int, seed: int):
        self._init()


class NextItemDataset(Dataset, MultiProcessDataLoaderSupport):
    def __init__(self, dataset: ItemSessionDataset, index: NextItemIndex):
        super(NextItemDataset, self).__init__()
        self._dataset = dataset
        self._index = index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        session_idx, target_pos = self._index[idx]
        session = self._dataset[session_idx][ITEM_SEQ_ENTRY_NAME]
        return {
            ITEM_SEQ_ENTRY_NAME: session[:target_pos],
            TARGET_ENTRY_NAME: session[target_pos]
        }

    def _mp_init(self, id: int, num_worker: int, seed: int):
        pass


class NextItemIterableDataset(IterableDataset, MultiProcessDataLoaderSupport):
    def __init__(self, dataset: ItemSessionDataset, index: NextItemIndex, seed: int = None):
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

    def _mp_init(self, id: int, num_worker: int, seed: int):
        # use evenly sized shards for each worker
        num_samples = len(self._index)
        self._start, self._stop = calculate_shard(id, num_worker, seed, num_samples)

    # FIXME (AD): we need to return some length, otherwise test does not work, see: https://github.com/PyTorchLightning/pytorch-lightning/issues/3500
    def __len__(self):
        return self._stop - self._start


def calculate_shard(id: int, num_worker: int, seed: int, num_samples: int) -> Tuple[int, int]:
    worker_share = int(math.ceil(num_samples / float(num_worker)))

    start = id * worker_share
    if id < num_worker - 1:
        #stop = min(start + worker_share - 1, num_samples)
        stop = min(start + worker_share, num_samples)
    else:
        stop = num_samples

    return start, stop
