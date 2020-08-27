import io
from pathlib import Path
import sys
from torch.utils.data import Dataset
from tqdm import tqdm

from data.datasets import ITEM_SEQ_ENTRY_NAME, INT_BYTE_SIZE, TARGET_ENTRY_NAME
from data.datasets.seqitem import SequentialItemSessionDataset


class NextItemPredSessionIndex:
    def __init__(self, dataset: SequentialItemSessionDataset, index_path: Path, min_session_length: int = 2, save_index: bool = True):

        if not index_path.exists() and save_index:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            self._create(dataset, index_path, min_session_length)

        self._index_file_handle = index_path.open("rb")

        self._min_session_length = self._read_min_session_length()
        self._length = self._read_length()

    def _create(self, dataset: SequentialItemSessionDataset, index_path: Path, min_session_length: int):
        current_idx = 0
        with index_path.open("wb") as index_file:
            for session_idx in tqdm(range(len(dataset)), desc="Creating Index."):
                sessions = dataset[session_idx][ITEM_SEQ_ENTRY_NAME]
                if len(sessions) > min_session_length:
                    for target_pos in range(1, len(sessions)):
                        self._write_entry(index_file, session_idx, target_pos)
                        current_idx += 1
            # write length at the end
            index_file.write(current_idx.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))
            # write minimum length
            index_file.write(min_session_length.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))

    @staticmethod
    def _write_entry(index_file, session_idx: int, target_pos: int):
        index_file.write(session_idx.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))
        index_file.write(target_pos.to_bytes(INT_BYTE_SIZE, byteorder=sys.byteorder, signed=False))

    def _read_length(self):
        self._index_file_handle.seek(-2*INT_BYTE_SIZE, io.SEEK_END)
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


class NextItemPredSessionDataset(Dataset):
    def __init__(self, dataset: SequentialItemSessionDataset, index: NextItemPredSessionIndex):
        super(NextItemPredSessionDataset, self).__init__()
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
