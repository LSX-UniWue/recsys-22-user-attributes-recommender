import typer
from pathlib import Path
from dataset.vocabulary import create_vocabulary

app = typer.Typer()


@app.command()
def build(data_file_path: Path = typer.Argument(..., exists=True, help="path to the input file in CSV format"),
          session_index_path: Path = typer.Argument(..., exists=True, help="path to the session index file"),
          vocabulary_output_file_path: Path = typer.Argument(..., exists=True, help='Output path for vocab file'),
          item_header_name: str = typer.Argument(..., help="name of the column that contains the item id"),
          delimiter: str = typer.Option("\t", help="the delimiter used in the CSV file.")) -> None:

    create_vocabulary.create_token_vocabulary(item_header_name=item_header_name,
                                              data_file_path=data_file_path,
                                              session_index_path=session_index_path,
                                              vocabulary_output_file_path=vocabulary_output_file_path,
                                              custom_tokens=[],
                                              delimiter=delimiter)
