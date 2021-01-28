import os
from pathlib import Path
from typing import List
from tqdm import tqdm

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.datasets.session import PlainSessionDataset, ItemSessionParser
from data.utils import create_indexed_header, read_csv_header
from tokenization.vocabulary import VocabularyBuilder, CSVVocabularyReaderWriter


def create_session_data_set(session_key: str, data_file_path: Path, index_file_path: Path,
                            delimiter: str) -> PlainSessionDataset:
    reader_index = CsvDatasetIndex(index_file_path)
    reader = CsvDatasetReader(data_file_path, reader_index)
    parser = ItemSessionParser(create_indexed_header(read_csv_header(data_file_path, delimiter=delimiter)),
                               session_key,
                               delimiter=delimiter)

    session_data_set = PlainSessionDataset(reader, parser)
    return session_data_set


def create_token_vocabulary(session_key: str, data_file_path: Path, session_index_path: Path,
                            vocabulary_output_path: Path,
                            custom_tokens: List[str], delimiter: str):
    vocab_builder = VocabularyBuilder()
    for token in custom_tokens:
        vocab_builder.add_token(token)

    data_set = create_session_data_set(session_key=session_key,
                                       data_file_path=data_file_path,
                                       index_file_path=session_index_path,
                                       delimiter=delimiter)

    for idx in tqdm(range(len(data_set)), desc=f"Tokenizing items from: {data_file_path}"):
        session = data_set[idx]
        session_tokens = session[ITEM_SEQ_ENTRY_NAME]

        for token in session_tokens:
            vocab_builder.add_token(token)

    vocabulary = vocab_builder.build()

    if not os.path.exists(vocabulary_output_path):
        vocabulary_output_path.mkdir(parents=True, exist_ok=True)
    with vocabulary_output_path.joinpath("tokens.txt").open("w") as file:
        CSVVocabularyReaderWriter().write(vocabulary, file)
