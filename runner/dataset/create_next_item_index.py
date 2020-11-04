from argparse import ArgumentParser
from pathlib import Path
from typing import Text

import typer

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets.nextitem import NextItemIndexBuilder
from data.datasets.session import ItemSessionDataset, ItemSessionParser
from data.utils import create_indexed_header, read_csv_header

app = typer.Typer()


@app.command()
def run(data: Path = typer.Argument(..., help="path to the input file in CSV format"),
        session_index: Path = typer.Argument(..., help="path to the session index file"),
        output: Path = typer.Argument(..., help="path to the output file"),
        item_header_name: Text = typer.Argument(..., help="name of the column that contains the item id"),
        min_session_length: int = typer.Option(2, help="the minimum acceptable session length"),
        delimiter: Text = typer.Option("\t", help="the delimiter used in the CSV file.")
        ):
    """
    Creates an index pointing to samples for the next item prediction task by adding a distinct sample for every
    target item in each session to the generated index.
    """

    session_index = CsvDatasetIndex(session_index)

    reader = CsvDatasetReader(data, session_index)
    session_parser = ItemSessionParser(
        create_indexed_header(read_csv_header(data, delimiter)),
        item_header_name,
        delimiter
    )
    dataset = ItemSessionDataset(reader, session_parser)
    builder = NextItemIndexBuilder(min_session_length)
    builder.build(dataset, output)


if __name__ == "__main__":
    app()
