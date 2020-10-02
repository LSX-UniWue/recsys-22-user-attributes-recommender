from typing import List

from numpy.random._generator import default_rng
from torch.utils.data import Dataset

from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.datasets.session import ItemSessionDataset
from tokenization.tokenizer import Tokenizer


class PosNegSessionDataset(Dataset):

    def __init__(self, dataset: ItemSessionDataset, tokenizer: Tokenizer, seed: int = None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self._rng = default_rng(seed=seed)

    def __getitem__(self, index: int):
        session = self.dataset[index][ITEM_SEQ_ENTRY_NAME]

        x = session[:-1]
        pos = session[1:]
        neg = self._sample_negative_target(session)

        return {
            "session": x,
            "positive_samples": pos,
            "negative_samples": neg
        }

    def __len__(self) -> int:
        return len(self.dataset)

    # TODO (AD) prevent sampler from generating special tokens, like <PAD>
    # We need to patch the tokenizer for this
    def _sample_negative_target(self, session) -> List[int]:
        tokens = set(self.tokenizer.get_vocabulary().ids())
        session = set(session)

        available_tokens = list(tokens - session)

        return self._rng.choice(available_tokens, len(session) - 1, replace=True).tolist()
