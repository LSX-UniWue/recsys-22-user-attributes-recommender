import sys
from pathlib import Path
from typing import Callable, Dict, Any, Iterable

from tqdm import tqdm

from data.datasets import ITEM_SEQ_ENTRY_NAME, INT_BYTE_SIZE
from data.datasets.session import ItemSessionDataset


class SessionPositionIndexBuilder:
    """
    This index builder builds a session position index. For each session returned by an ItemSessionDataset,
    this builder can generate m different positions for the session.
    For example you will test every position of a session, you return a range from 1 to the session length.
    Than the session will be used session length - 1 times with different targets
    """

    def __init__(self,
                 min_session_length: int = 2,
                 target_positions_extractor: Callable[[Dict[str, Any]], Iterable[int]] = None
                 ):
        self._min_session_length = min_session_length
        if target_positions_extractor is None:
            target_positions_extractor = all_remaining_items
        self._target_positions_extractor = target_positions_extractor

    def build(self,
              dataset: ItemSessionDataset,
              index_path: Path
              ):
        if not index_path.exists():
            index_path.parent.mkdir(parents=True, exist_ok=True)

        current_idx = 0
        with index_path.open("wb") as index_file:
            for session_idx in tqdm(range(len(dataset)), desc="Creating Index."):
                session = dataset[session_idx]
                items = session[ITEM_SEQ_ENTRY_NAME]
                # remove session with lower min session length
                if len(items) > self._min_session_length:
                    target_positions = self._target_positions_extractor(session)
                    for target_pos in target_positions:
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


def all_remaining_items(session: Dict[str, Any]
                        ) -> Iterable[int]:
    return range(1, len(session[ITEM_SEQ_ENTRY_NAME]))
