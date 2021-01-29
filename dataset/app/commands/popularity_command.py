import typer
from pathlib import Path
from dataset.popularity import build_popularity

app = typer.Typer()


@app.command()
def build(
        data_file_path: Path = typer.Argument(..., exists=True, help="path to the input file in CSV format"),
        session_index_path: Path = typer.Argument(..., exists=True, help="path to the session index file"),
        vocabulary_file_path: Path = typer.Argument(..., exists=True, help='path to the vocab file'),
        output_file_path: Path = typer.Argument(..., help="path to the output file"),
        item_header_name: str = typer.Argument(..., help="name of the column that contains the item id"),
        min_session_length: int = typer.Option(2, help="the minimum acceptable session length"),
        delimiter: str = typer.Option("\t", help="the delimiter used in the CSV file.")) -> None:

    build_popularity.build(data_file_path=data_file_path, session_index_path=session_index_path,
                           vocabulary_file_path=vocabulary_file_path, output_file_path=output_file_path,
                           item_header_name=item_header_name, min_session_length=min_session_length,
                           delimiter=delimiter)
