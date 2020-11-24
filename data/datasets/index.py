import io
import sys
from pathlib import Path

from data.datasets import INT_BYTE_SIZE
from data.mp import MultiProcessSupport


class SessionPositionIndex(MultiProcessSupport):

    def __init__(self,
                 index_path: Path
                 ):
        super().__init__()
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

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        self._init()
