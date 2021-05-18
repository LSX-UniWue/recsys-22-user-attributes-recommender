from abc import abstractmethod
from typing import Dict, Any, Iterable

from data.datasets import ITEM_SEQ_ENTRY_NAME


class TargetPositionExtractor:

    @abstractmethod
    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        pass

    def __call__(self, session: Dict[str, Any]) -> Iterable[int]:
        return self.apply(session)


class FixedOffsetPositionExtractor(TargetPositionExtractor):

    def __init__(self, offset: int):
        self.offset = offset

    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        return [len(session[ITEM_SEQ_ENTRY_NAME]) - self.offset]


class RemainingSessionPositionExtractor(TargetPositionExtractor):

    def __init__(self, min_session_length: int = 1):
        self.min_session_length = min_session_length

    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        return range(self.min_session_length, len(session[ITEM_SEQ_ENTRY_NAME]))