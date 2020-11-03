from typing import List

from numpy.random._generator import default_rng
from torch.utils.data import Dataset

from data.datasets import ITEM_SEQ_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME
from data.datasets.session import ItemSessionDataset
from tokenization.tokenizer import Tokenizer


class PosNegSessionDataset(Dataset):

    def __init__(self, dataset: ItemSessionDataset, tokenizer: Tokenizer, seed: int = None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self._rng = default_rng(seed=seed)

    def __getitem__(self, index: int):
        session = self.dataset[index][ITEM_SEQ_ENTRY_NAME]

        assert(len(session) > 1)

        x = session[:-1]
        pos = session[1:]
        neg = self._sample_negative_target(session)

        # TODO (AD) right now we only support ranking the full item set. Often on datasets with large item spaces,
        #  only a sample is drawn and ranked. We need to parameterize this class so that different strategies can be
        #  used.
        # FIXME: this should only add the items to the map, not create a new one
        return {
            ITEM_SEQ_ENTRY_NAME: x,
            POSITIVE_SAMPLES_ENTRY_NAME: pos,
            NEGATIVE_SAMPLES_ENTRY_NAME: neg,
            "items": self.tokenizer.get_vocabulary().ids() # use the full dataset for ranking (for now)
        }

    def __len__(self) -> int:
        return len(self.dataset)

    # TODO (AD) prevent sampler from generating special tokens, like <PAD>
    # We need to patch the tokenizer for this
    def _sample_negative_target(self, session) -> List[int]:
        tokens = set(self.tokenizer.get_vocabulary().ids())
        used_tokens = set(session)

        available_tokens = list(tokens - used_tokens)

        return self._rng.choice(available_tokens, len(session) - 1, replace=True).tolist()
