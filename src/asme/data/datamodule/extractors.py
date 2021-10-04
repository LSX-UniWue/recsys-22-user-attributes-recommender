from abc import abstractmethod
from typing import Dict, Any, Iterable, Optional

from asme.data.datasets import ITEM_SEQ_ENTRY_NAME


class TargetPositionExtractor:
    """
    Base class for different strategies regarding the extraction of target indices from a session.
    """

    @abstractmethod
    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        """
        Extracts a set of target indices from the session.

        :param session: A dictionary containing the information of a single session.
        """
        pass

    def __call__(self, session: Dict[str, Any]) -> Iterable[int]:
        return self.apply(session)


class FixedOffsetPositionExtractor(TargetPositionExtractor):
    """
    This TargetPositionExtractor always returns the index exactly offset positions away from the end of the session.
    """

    def __init__(self, offset: int):
        """
        :param offset: The offset of the target position from the end of the sequence. This offset ranges from 1 for the
        last item in the sequence to sequence_length for the second item in the sequence.
        """
        self.offset = offset

    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        return [len(session[ITEM_SEQ_ENTRY_NAME]) - self.offset - 1]


class RemainingSessionPositionExtractor(TargetPositionExtractor):
    """
    This TargetPositionExtractor returns all indices between min_session_length and sequence_length
    """

    def __init__(self, min_session_length: int = 1):
        """
        :param min_session_length: The minimum length of a subsequence of the session that has to be reached before
        extracting targets begins. For instance, min_session_length = 5 would yield 5 (i.e the sixth entry) as the first
        target index.
        """
        self.min_session_length = min_session_length

    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        return range(self.min_session_length, len(session[ITEM_SEQ_ENTRY_NAME]))


class SlidingWindowPositionExtractor(TargetPositionExtractor):
    """
    The SlidingWindowPositionExtractor returns all indices between [min_input_length; len(sequence) - session_end_offset]

    Example:
    session: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    window_size: 5, which is computed by the sum of the sequence_length and target_length
    Since there often is more than one target per "slide", we return the position of the last target.
    The session_offset gibt an, wieviele Elemente von rechts nicht als targets betrachtet werden. (LOO)
    Also since we only use step_size=1 when sliding over the sequence with the window, we return a range.
    Die min_input_length als optionaler Parameter wird verwendet, falls
    Aber wie gehen wir mit den jeweiligen Größen und Padding um?
    -> Vergleich mit Caser/Cosrec
    """
    def __init__(self, window_size: int, session_end_offset: int, min_input_length: Optional[int] = None):
        self.window_size = window_size
        self.session_end_offset = session_end_offset

        if min_input_length is None:
            self.min_input_length = self.window_size - 1
        else:
            self.min_input_length = min_input_length

    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        sequence = session[ITEM_SEQ_ENTRY_NAME]
        return range(self.min_input_length, len(sequence) - self.session_end_offset)
