from pathlib import Path
from typing import Tuple, List
import sys
import io

from data.mp import MultiProcessDataLoaderSupport


class CsvDatasetIndex(MultiProcessDataLoaderSupport):

    INT_BYTE_SIZE = 8

    def __init__(self, index_file_path: Path):
        self._index_file_path = index_file_path

        self._init()

    def _init(self):
        self._index_file_handle = self._index_file_path.open("rb")
        self._num_sessions = self._read_num_sessions()

    def _read_num_sessions(self) -> int:
        self._seek_to_header()
        return self._read_int()

    def _seek_to_header(self):
        self._index_file_handle.seek(-self.INT_BYTE_SIZE, io.SEEK_END)

    def _read_int(self) -> int:
        return int.from_bytes(self._index_file_handle.read(self.INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

    def num_sessions(self) -> int:
        return self._num_sessions

    def __enter__(self):
        return self._index_file_handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._index_file_handle.close()

    def __len__(self):
        return self.num_sessions()

    def get(self, session_num: int) -> Tuple[int, int]:
        """
        Returns the boundaries of a specific session as byte positions within the file. The sessions are sequentially
        numbered and 0-based.

        :param session_num: a number in [0; num_sessions-1]
        :return: a tuple with start and end byte position within the data file.
        """
        self._index_file_handle.seek(session_num * self.INT_BYTE_SIZE * 2)

        start = int.from_bytes(self._index_file_handle.read(self.INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)
        end = int.from_bytes(self._index_file_handle.read(self.INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

        return start, end

    def _mp_init(self, id: int, num_worker: int, seed: int):
        self._init()


class CsvDatasetReader(MultiProcessDataLoaderSupport):
    def __init__(self, data_file_path: Path, index: CsvDatasetIndex):
        self._data_file_path = data_file_path
        self._index = index

        self._init()

    def _init(self):
        self._data_file_handle = self._data_file_path.open(mode="rb")
        self._num_sessions = self._index.num_sessions()

    def __len__(self):
        return self._num_sessions

    def __enter__(self):
        return self._data_file_handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._data_file_handle.close()

    def get_session(self, idx: int) -> str:
        if idx >= self._num_sessions:
            raise Exception(f"{idx} is not a valid index in [0, {self._num_sessions}]")

        start, end = self._index.get(idx)
        session_raw = self._read_session(start, end)

        return session_raw

    def _read_session(self, start: int, end: int) -> str:
        self._data_file_handle.seek(start)
        return self._data_file_handle.read(end - start).decode(encoding="utf-8")

    def _mp_init(self, id: int, num_worker: int, seed: int):
        self._init()
