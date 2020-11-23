from pathlib import Path
from typing import List

import typer

from data.base.indexer import CsvSessionIndexer

app = typer.Typer()


@app.command()
def create_index_for_csv(input_path: Path = typer.Argument(..., exists=True, help="file with sessions"),
                         output_path: Path = typer.Argument(..., help='index file'),
                         session_key: List[str] = typer.Argument(..., help='the names of the columns that comprise the'
                                                                           'session key'),
                         delimiter: str = typer.Option('\t', help='delimiter for the csv file')):

    index = CsvSessionIndexer(delimiter=delimiter)
    index.create(input_path, output_path, session_key)


if __name__ == "__main__":
    app()
