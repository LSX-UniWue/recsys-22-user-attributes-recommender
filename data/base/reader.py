from pathlib import Path
from typing import Tuple
import sys
import io

from data.mp import MultiProcessSupport


class CsvDatasetIndex(MultiProcessSupport):

    INT_BYTE_SIZE = 8

    def __init__(self, index_file_path: Path):
        self._index_file_path = index_file_path

        self._init()

    def _init(self):
        with self._index_file_path.open("rb") as file_handle:
            self._num_sessions = self._read_num_sessions(file_handle)

    def _read_num_sessions(self, file_handle) -> int:
        self._seek_to_header(file_handle)
        return self._read_int(file_handle)

    def _seek_to_header(self, file_handle):
        file_handle.seek(-self.INT_BYTE_SIZE, io.SEEK_END)

    def _read_int(self, file_handle) -> int:
        return int.from_bytes(file_handle.read(self.INT_BYTE_SIZE), byteorder=sys.byteorder, signed=False)

    def num_sessions(self) -> int:
        return self._num_sessions

    def __len__(self):
        return self.num_sessions()

    def get(self, session_num: int) -> Tuple[int, int]:
        """
        Returns the boundaries of a specific session as byte positions within the file. The sessions are sequentially
        numbered and 0-based.

        :param session_num: a number in [0; num_sessions-1]
        :return: a tuple with start and end byte position within the data file.
        """
        with self._index_file_path.open("rb") as file_handle:
            file_handle.seek(session_num * self.INT_BYTE_SIZE * 2)

            start = self._read_int(file_handle)
            end = self._read_int(file_handle)

            return start, end

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        self._init()


class CsvDatasetReader(MultiProcessSupport):
    def __init__(self,
                 data_file_path: Path,
                 index: CsvDatasetIndex
                 ):
        self._data_file_path = data_file_path
        self._index = index

        self._init()

    def _init(self):
        self._num_sessions = self._index.num_sessions()

    def __len__(self):
        return self._num_sessions

    def get_session(self, idx: int) -> str:
        if idx >= self._num_sessions:
            raise Exception(f"{idx} is not a valid index in [0, {self._num_sessions}]")

        start, end = self._index.get(idx)
        session_raw = self._read_session(start, end)

        return session_raw

    def _read_session(self, start: int, end: int) -> str:
        with self._data_file_path.open("rb") as file_handle:
            file_handle.seek(start)
            return file_handle.read(end - start).decode(encoding="utf-8")

    def _init_class_for_worker(self, worker_id: int, num_worker: int, seed: int):
        self._init()
