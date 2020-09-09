from argparse import ArgumentParser
from pathlib import Path
from typing import Text

from data.base.reader import CsvSessionDatasetIndex, CsvSessionDatasetReader
from data.datasets.nextitem import NextItemIndexBuilder
from data.datasets.session import ItemSessionDataset, ItemSessionParser
from data.utils import create_indexed_header, read_csv_header


def run(data_file_path: Path, session_index_path, output_file_path: Path, item_header_name: Text, min_session_length: int = 2, delimiter: Text = "\t"):
    session_index = CsvSessionDatasetIndex(session_index_path)
    reader = CsvSessionDatasetReader(data_file_path, session_index)
    session_parser = ItemSessionParser(
        create_indexed_header(read_csv_header(data_file_path, delimiter)),
        item_header_name,
        delimiter
    )
    dataset = ItemSessionDataset(reader, session_parser)

    builder = NextItemIndexBuilder(min_session_length)
    builder.build(dataset, output_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_file", help="path to the csv data file", type=str)
    parser.add_argument("session_index_file", help="path to the session index file", type=str)
    parser.add_argument("output_file", help="path where the index is written to", type=str)
    parser.add_argument("item_header_name", help="name of the item column (header)", type=str)
    parser.add_argument("--min_session_length", help="minimum session length (Default: 2)", default=2, type=int)
    parser.add_argument("--delimiter", help="delimiter used in the csv data file (Default: \\t).", default="\t", type=str)

    args = parser.parse_args()

    data_file = Path(args.data_file)
    session_index_file = Path(args.session_index_file)
    output_file = Path(args.output_file)

    item_header_name = args.item_header_name
    min_session_length = args.min_session_length
    delimiter = args.delimiter

    run(data_file, session_index_file, output_file, item_header_name, min_session_length, delimiter)
