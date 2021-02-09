import typer
from pathlib import Path

from dataset.app.index_command import index_csv, create_conditional_index
from dataset.dataset_index_splits.ratio_split import run as create_ratio_splits

app = typer.Typer()


@app.command()
def create_splits(dataset: str = typer.Argument(..., help="ml-1m or ml-20m"),
                  session_key: str = typer.Argument("userId", help="session key"),
                  item_header: str = typer.Argument("title", help="item column"),
                  output_dir: Path = typer.Option("./dataset/", help='directory to save data'),
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
    create_ratio_splits(path_main_csv, path_main_index, split_dir_path, splits, seed)

    for split in ["test", "train", "valid"]:
        split_path = split_dir_path / f'{split}.csv'
        split_path_index = split_dir_path / f'{split}.idx'
        split_path_next_index = split_dir_path / f'{split}.nip'
        index_csv(split_path, split_path_index, session_key=[session_key])
        create_conditional_index(data_file_path=split_path,
                                 session_index_path=split_path_index,
                                 output_file_path=split_path_next_index,
                                 item_header_name=item_header,
                                 min_session_length=min_session_length,
                                 delimiter="\t")


if __name__ == "__main__":
    app()
