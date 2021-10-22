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

    :param window_size: The size of each "window" of the session. Each window consists of a sequence
                        order and number of targets.
    :param session_end_offset: Indicates how many items are cut off from the right. Used for leave one out

    Example:
    session: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    window_size: 5, session_offset: 2

    sequences would be (started from the right): [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8]

    -> we need the position of each last target:
    so the position of 5, 6, 7, 8 which is index 4 - 7 -> windowsize-1 to sequence_length-session_offset

    That means we return the range between 4-7. With that information, the window size and the target size of each window
    we can reconstruct each window easily.

    """
    def __init__(self, window_markov_length: int, window_target_length: int, session_end_offset: int, ):
        self.window_size = window_markov_length + window_target_length
        self.session_end_offset = session_end_offset

    def apply(self, session: Dict[str, Any]) -> Iterable[int]:
        sequence = session[ITEM_SEQ_ENTRY_NAME]
        return range(self.window_size - 1, len(sequence) - self.session_end_offset)
