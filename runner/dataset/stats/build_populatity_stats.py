from pathlib import Path

import typer
from tqdm import tqdm

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.datasets.session import PlainSessionDataset, ItemSessionDataset, ItemSessionParser
from data.utils import read_csv_header, create_indexed_header
from tokenization.tokenizer import Tokenizer
from tokenization.vocabulary import CSVVocabularyReaderWriter

app = typer.Typer()


@app.command()
def build_popularity_stats(data_file_path: Path = typer.Argument(..., exists=True,
                                                                 help="path to the input file in CSV format"),
                           session_index_path: Path = typer.Argument(..., exists=True,
                                                                     help="path to the session index file"),
                           vocabulary_file_path: Path = typer.Argument(..., exists=True, help='path to the vocab file'),
                           output_file_path: Path = typer.Argument(..., help="path to the output file"),
                           item_header_name: str = typer.Argument(...,
                                                                  help="name of the column that contains the item id"),
                           min_session_length: int = typer.Option(2, help="the minimum acceptable session length"),
                           delimiter: str = typer.Option("\t", help="the delimiter used in the CSV file."),
                           ) -> None:
    session_parser = ItemSessionParser(
        create_indexed_header(read_csv_header(data_file_path, delimiter)),
        item_header_name,
        delimiter=delimiter
    )

    session_index = CsvDatasetIndex(session_index_path)
    reader = CsvDatasetReader(data_file_path, session_index)

    plain_dataset = PlainSessionDataset(reader, session_parser)
    dataset = ItemSessionDataset(plain_dataset)

    # load the tokenizer
    vocabulary_reader = CSVVocabularyReaderWriter()
    vocabulary = vocabulary_reader.read(vocabulary_file_path.open("r"))
    tokenizer = Tokenizer(vocabulary)

    counts = {}

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

    # write to file
    with open(output_file_path, 'w') as output_file:
        # loop through the vocab to also get the special tokens
        for token_id, _ in vocabulary.id_to_token.items():
            count = counts.get(token_id, 0)
            output_file.write(f"{count / total_count}\n")


if __name__ == "__main__":
    app()
