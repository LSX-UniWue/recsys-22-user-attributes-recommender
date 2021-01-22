from typing import Union, List, Dict, Any, Set

from numpy.random._generator import default_rng

from data.datasets import ITEM_SEQ_ENTRY_NAME, SAMPLE_IDS, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME
from data.datasets.processors.processor import Processor
from tokenization.tokenizer import Tokenizer


class PositiveNegativeSamplerProcessor(Processor):

    def __init__(self,
                 tokenizer: Tokenizer,
                 seed: int = 42
                 ):
        super().__init__()
        self._tokenizer = tokenizer
        self._rng = default_rng(seed=seed)

    def _sample_negative_target(self,
                                session: Union[List[int], List[List[int]]]
                                ) -> Union[List[int], List[List[int]]]:
        # get all possible tokens
        tokens = set(self._tokenizer.get_vocabulary().ids())
        # remove special tokens TODO: maybe move to tokenizer?
        tokens = tokens - set(self._tokenizer.get_special_token_ids())
        used_tokens = _get_all_tokens_of_session(session)

        available_tokens = list(tokens - used_tokens)

        if isinstance(session[0], list):
            results = []
            for seq_step in session[: -1]:  # skip last target
                neg_samples = self._rng.choice(available_tokens, len(seq_step), replace=True).tolist()
                results.append(neg_samples)

            return results

        return self._rng.choice(available_tokens, len(session) - 1, replace=True).tolist()

    def _get_all_tokens_of_session(self, session: Union[List[int], List[List[int]]]
                                   ) -> Set[int]:
        if isinstance(session[0], list):
            flat_items = [item for sublist in session for item in sublist]
        else:
            flat_items = session

        return set(flat_items)

    def process(self, parsed_session: Dict[str, Any]) -> Dict[str, Any]:
        session = parsed_session[ITEM_SEQ_ENTRY_NAME]

        if len(session) == 1:
            raise AssertionError(f'{parsed_session[SAMPLE_IDS]} : {parsed_session["pos"]}')

        x = session[:-1]
        pos = session[1:]
        neg = self._sample_negative_target(session)

        parsed_session[ITEM_SEQ_ENTRY_NAME] = x
        parsed_session[POSITIVE_SAMPLES_ENTRY_NAME] = pos
        parsed_session[NEGATIVE_SAMPLES_ENTRY_NAME] = neg
        return parsed_session
