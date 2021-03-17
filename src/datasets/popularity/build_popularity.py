import os
from tqdm import tqdm
from pathlib import Path
from typing import Dict

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.datasets.sequence import PlainSequenceDataset, ItemSequenceDataset, ItemSessionParser
from data.utils import read_csv_header, create_indexed_header
from asme.tokenization.tokenizer import Tokenizer
from asme.tokenization.vocabulary import CSVVocabularyReaderWriter


def build(data_file_path: Path, session_index_path: Path, vocabulary_file_path: Path, output_file_path: Path,
          item_header_name: str, min_session_length: int, delimiter: str) -> None:
    """
    Builds the popularity distribution of the items in the data set. This enables us to later sample the items
    based on their original distribution, e.g., for evaluation.

    :param data_file_path: CSV file containing original data
    :param session_index_path: index file belonging to the data file
    :param vocabulary_file_path: vocabulary file belonging to the data file
    :param output_file_path: output file where the popularity should be written to
    :param item_header_name: Name of the item key in the data set, e.g, "ItemId"
    :param min_session_length: minimum session length determining which sessions should be used
    :param delimiter: delimiter used in data file
    :return: None, Side Effect: popularity distribution is written to output_file_path
    """
    session_parser = ItemSessionParser(
        create_indexed_header(read_csv_header(data_file_path, delimiter)),
        item_header_name,
        delimiter=delimiter
    )

    session_index = CsvDatasetIndex(session_index_path)
    reader = CsvDatasetReader(data_file_path, session_index)

    plain_dataset = PlainSequenceDataset(reader, session_parser)
    dataset = ItemSequenceDataset(plain_dataset)

    # load the tokenizer
    vocabulary_reader = CSVVocabularyReaderWriter()
    vocabulary = vocabulary_reader.read(vocabulary_file_path.open("r"))
    tokenizer = Tokenizer(vocabulary)

    counts: Dict[str, int] = {}
    for session_idx in tqdm(range(len(dataset)), desc="Counting items"):
        session = dataset[session_idx]
        items = session[ITEM_SEQ_ENTRY_NAME]
        # ignore session with lower min session length
        if len(items) > min_session_length:
            converted_tokens = tokenizer.convert_tokens_to_ids(items)
            for token in converted_tokens:
                count = counts.get(token, 0)
                count += 1
                counts[token] = count
    total_count = sum(counts.values())
    # print("Total count", total_count)
    # print("Counts dict", counts)
    if not os.path.exists(output_file_path.parent):
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
    # write to file
    with output_file_path.open('w') as output_file:
        # loop through the vocab to also get the special tokens
        for token_id, _ in vocabulary.id_to_token.items():
            count = counts.get(token_id, 0)
            output_file.write(f"{count / total_count}\n")
