import collections
from typing import Any, List


SPECIAL_TOKENS_ATTRIBUTES = ["bos_token", "eos_token", "unk_token", "sep_token",
                             "pad_token", "cls_token", "mask_token"]


class PreTrainedItemizer(object):

    def __init__(self, max_len=None, **kwargs):
        self.bos_token = None
        self.eos_token = None
        self.unk_token = None
        self.mask_token = None
        self.pad_token = None
        self.sep_token = None
        self.cls_token = None

        self._additional_special_tokens = []

        self.max_len = max_len if max_len is not None else int(1e12)

        # Added tokens
        self.added_tokens_encoder = {}
        self.added_tokens_decoder = {}

        # inputs and kwargs for saving and re-loading (see ``from_pretrained`` and ``save_pretrained``)
        self.init_inputs = ()
        self.init_kwargs = {}

        for key, value in kwargs.items():
            if key in SPECIAL_TOKENS_ATTRIBUTES:
                assert isinstance(value, str)
                setattr(self, key, value)

    @property
    def pad_token_id(self) -> int:
        return self.convert_items_to_ids(self.pad_token)

    @property
    def sep_token_id(self) -> int:
        return self.convert_items_to_ids(self.sep_token)

    @property
    def cls_token_id(self) -> int:
        return self.convert_items_to_ids(self.cls_token)

    def convert_items_to_ids(self,
                             items: Any) -> Any:
        if items is None:
            return None

        if isinstance(items, str):
            return self._convert_items_to_id_with_added_voc(items)

        ids = []
        for item in items:
            ids.append(self._convert_items_to_id_with_added_voc(item))
        return ids

    def _convert_items_to_id_with_added_voc(self, item):
        if item is None:
            return None

        if item in self.added_tokens_encoder:
            return self.added_tokens_encoder[item]
        return self._convert_item_to_id(item)

    def get_special_tokens_mask(self,
                                item_ids: List[int],
                                second_item_ids: List[int] = None,
                                already_has_special_tokens: bool = False) -> List[bool]:
        if already_has_special_tokens:
            if second_item_ids is not None:
                raise ValueError("You should not supply a second sequence if the provided sequence of "
                                 "ids is already formated with special tokens for the model.")
            return list(map(lambda x: x in [self.sep_token_id, self.cls_token_id], item_ids))

        if second_item_ids is None:
            return [True] + ([False] * len(item_ids)) + [True]
        return [True] + ([False] * len(item_ids)) + [True, True] + ([False] * len(second_item_ids)) + [True]

    def _convert_item_to_id(self, item):
        raise NotImplemented('please implement the item to id lookup in the itemizer')

    @property
    def vocab_size(self) -> int:
        """ Size of the base vocabulary (without the added tokens) """
        raise NotImplementedError

    def __len__(self):
        """ Size of the full vocabulary with the added tokens """
        return self.vocab_size + len(self.added_tokens_encoder)


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab