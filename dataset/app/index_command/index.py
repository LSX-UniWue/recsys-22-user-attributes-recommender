import typer
from pathlib import Path
from typing import List
from data.base.indexer import CsvSessionIndexer
from dataset.app.app import get_dataset_app

app = get_dataset_app()


@app.command()
def index(data_file_path: Path = typer.Argument(..., exists=True, help="file with sessions"),
          index_file_path: Path = typer.Argument(..., help='index file'),
          session_key: List[str] = typer.Argument(..., help='the names of the columns that comprise the'
                                                            'session key'),
          delimiter: str = typer.Option('\t', help='delimiter for the csv file')):
    """
    :param data_file_path: Path to the CSV-file containing sessions that is to be indexed
    :param index_file_path: Path that the index file should be written to
    :param session_key: Dataset key under which the session IDs are stored
    :param delimiter: delimiter used in the CSV
    :return: Writes index file for .csv-file in input path to output path
    """
    csv_index = CsvSessionIndexer(delimiter=delimiter)
    csv_index.create(data_file_path, index_file_path, session_key)

