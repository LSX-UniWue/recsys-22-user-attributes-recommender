import io
from typing import Dict, List, Any, Text
import csv
from torch.utils.data import Dataset

from data.base.reader import CsvSessionDatasetReader
from data.datasets import ITEM_SEQ_ENTRY_NAME
from tokenization.tokenizer import Tokenizer


class SessionParser(object):
    def parse(self, raw_session: Text) -> Dict[Text, List[Any]]:
        raise NotImplementedError()


class SequentialItemSessionParser(object):

    def __init__(self, indexed_headers: Dict[Text, int], item_header_name: Text, delimiter: Text = "\t"):
        self._indexed_headers = indexed_headers
        self._item_header_name = item_header_name
        self._delimiter = delimiter

    def parse(self, raw_session: Text) -> Dict[Text, List[Any]]:
        reader = csv.reader(io.StringIO(raw_session), delimiter=self._delimiter)
        items = [self._get_item(entry) for entry in reader]
        return {
            ITEM_SEQ_ENTRY_NAME: items
        }

    def _get_item(self, entry: List[Text]) -> int:
        item_column_idx = self._indexed_headers[self._item_header_name]
        return int(entry[item_column_idx])


class SequentialItemSessionDataset(Dataset):
    def __init__(self, reader: CsvSessionDatasetReader, parser: SessionParser, itemizer: Tokenizer = None):
        self._reader = reader
        self._parser = parser
        self._itemizer = itemizer

    def __len__(self):
        return len(self._reader)

    def __getitem__(self, idx):
        items = self._parser.parse(self._reader.get_session(idx))[ITEM_SEQ_ENTRY_NAME]

        if self._itemizer:
            itemized_items = self._itemizer.convert_tokens_to_ids(items)
        else:
            itemized_items = items

        return {
            ITEM_SEQ_ENTRY_NAME: itemized_items
        }

