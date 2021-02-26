import typer
from pathlib import Path

from dataset.app.index_command import index_csv
from dataset.dataset_index_splits.ratio_split import run as create_ratio_splits

app = typer.Typer()


@app.command()
def create_splits(dataset: str = typer.Argument(..., help="ml-1m or ml-20m"),
                  session_key: str = typer.Argument("userId", help="session key"),
                  item_header: str = typer.Argument("title", help="item column"),
                  output_dir: Path = typer.Option("./dataset/", help='directory to save data'),
                  delimiter: str = typer.Option("\t", help="the delimiter used in the CSV file."),
                  seed: int = typer.Option(123456, help='seed for split'),
                  train: float = typer.Option(0.9, help="train_split"),
                  valid: float = typer.Option(0.05, help="train_split"),
                  test: float = typer.Option(0.05, help="train_split"),
                  min_session_length: int = typer.Option(2, help="minimum session length")
                  ) -> None:
    """
    FixMe I need documentation
    :param dataset:
    :param session_key:
    :param item_header:
    :param output_dir:
    :param delimiter
    :param seed:
    :param train:
    :param valid:
    :param test:
    :param min_session_length:
    :return:
    """

    dataset_dir = output_dir / dataset
    path_main_csv = dataset_dir / f'{dataset}.csv'
    path_main_index = dataset_dir / f'{dataset}.idx'
    split_dir_path = dataset_dir / 'splits'
    split_dir_path.mkdir(parents=True, exist_ok=True)

    index_csv(path_main_csv, path_main_index, session_key=[session_key])

    splits = {"train": train, "valid": valid, "test": test}
    create_ratio_splits(data_file_path=path_main_csv, match_index_path=path_main_index, output_dir_path=split_dir_path,
                        split_ratios=splits, delimiter=delimiter, session_key=[session_key],
                        item_header_name=item_header, minimum_session_length=min_session_length, seed=seed)


if __name__ == "__main__":
    app()
