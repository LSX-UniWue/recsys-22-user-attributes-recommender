import typer
from pathlib import Path

from datasets.app.index_command import index_csv
from datasets.dataset_index_splits.strategy_split import run_strategy_split
from datasets.data_structures.DatasetMetadata import DatasetMetadata
from datasets.data_structures.SplitStrategy import SplitStrategy
from datasets.dataset_index_splits import SplitStrategiesFactory

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

    ratio_split_strategy: SplitStrategy = SplitStrategiesFactory.get_ratio_strategy(train_ratio=train,
                                                                                    validation_ratio=valid,
                                                                                    test_ratio=test,
                                                                                    seed=seed)
    dataset_metadata: DatasetMetadata = DatasetMetadata(
        data_file_path=path_main_csv,
        session_index_path=path_main_index,
        session_key=[session_key],
        delimiter=delimiter,
        item_header_name=item_header
    )
    run_strategy_split(dataset_metadata=dataset_metadata,
                       output_dir_path=split_dir_path,
                       split_strategy=ratio_split_strategy,
                       minimum_session_length=min_session_length)


if __name__ == "__main__":
    app()
