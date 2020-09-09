from argparse import ArgumentParser
from pathlib import Path
from typing import Text

from data.base.indexer import CsvSessionIndexer


def run(input_path: Path, output_path: Path, session_key: Text, delimiter: Text = "\t"):

    index = CsvSessionIndexer(delimiter=delimiter)
    index.create(input_path, output_path, session_key)


if __name__ == "__main__":
    parser = ArgumentParser(prog="create_index")
    parser.add_argument("input", help="file with sessions")
    parser.add_argument("output", help="index file")
    parser.add_argument("--delimiter", help="delimiter for the csv file", default="\t")
    parser.add_argument("--session_key", nargs="+", help="the names of the columns that comprise the session key")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    session_key = args.session_key
    delimiter = args.delimiter

    run(input_path, output_path, session_key, delimiter)
