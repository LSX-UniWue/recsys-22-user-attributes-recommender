import collections
from typing import Any, List, Optional, OrderedDict, TextIO
import csv

SPECIAL_TOKENS_ATTRIBUTES = ["bos_token", "eos_token", "unk_token", "sep_token",
                             "pad_token", "cls_token", "mask_token"]


class Vocabulary(object):
    def __init__(self, token_to_id: OrderedDict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = collections.OrderedDict([(id, token) for token, id in token_to_id.items()])

    def get_id(self, token: str) -> Optional[int]:
        if token not in self.token_to_id:
            return None

        return self.token_to_id[token]

    def get_token(self, id: int) -> Optional[str]:
        if id not in self.id_to_token:
            return None

        return self.id_to_token[id]

    def tokens(self) -> List[str]:
        return list(map(lambda x: x[0], self.token_to_id.items()))

    def ids(self) -> List[int]:
        return list(map(lambda x: x[1], self.token_to_id.items()))

    def __len__(self):
        return len(self.token_to_id)


class VocabularyBuilder(object):
    def __init__(self, tokens: List[str] = None, start_id: int = 0):
        self.next_id = start_id + len(tokens)
        self.token_to_id = collections.OrderedDict(zip(tokens, range(start_id, self.next_id)))

    def add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = self.next_id
            self.next_id += 1

        return self.token_to_id[token]

    def build(self) -> Vocabulary:
        return Vocabulary(self.token_to_id)


class VocabularyReaderWriter(object):
    def write(self, vocabulary: Vocabulary, output: TextIO):
        raise NotImplementedError()

    def read(self, input: TextIO) -> Vocabulary:
        raise NotImplementedError()


class CSVVocabularyReaderWriter(VocabularyReaderWriter):
    """
        Encodes the vocabulary as a CSV data file with every entry following the pattern: <token><delimiter><id>.
    """
    def __init__(self, delimiter: str = "\t"):
        self.delimiter = delimiter

    def write(self, vocabulary: Vocabulary, output: TextIO):
        writer = csv.writer(output, delimiter=self.delimiter)
        for token in vocabulary.tokens():
            writer.writerow([token, vocabulary.get_id(token)])

    def read(self, input: TextIO) -> Vocabulary:
        reader = csv.reader(input, delimiter=self.delimiter)
        vocabulary_entries = [(token, id) for [token, id] in reader]

        return Vocabulary(collections.OrderedDict(vocabulary_entries))


class SequentialIdVocabularyReaderWriter(VocabularyReaderWriter):
    """
        Assumes that the vocabulary consists of consecutively numbered tokens starting with 0. Only writes the tokens
        in this order and recovers the ids on reading by assuming that the id is identical to the line number of the
        token.
    """
    def write(self, vocabulary: Vocabulary, output: TextIO):
        for token in vocabulary.tokens():
            output.write(token)
            output.write("\n")

    def read(self, input: TextIO) -> Vocabulary:
        tokens = [token.strip() for token in input]
        token_to_id = zip(tokens, range(len(tokens)))

        return Vocabulary(collections.OrderedDict(token_to_id))


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