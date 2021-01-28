import typer
from typing import Dict, Any, Iterable
from pathlib import Path
from dataset.dataset_splits import next_item_split, ratio_split

app = typer.Typer()


@app.command()
def next_item(
        data_file_path: Path = typer.Argument(..., help="Data file in csv format"),
        session_index_path: Path = typer.Argument(..., help="Path to session index for the data file"),
        output_dir_path: Path = typer.Argument(..., help="path that the splits should be written to"),
        minimum_session_length: int = typer.Option(4, help="Minimum length of sessions that are to be included"),
        delimiter: str = typer.Option('\t', help="Delimiter used in data file"),
        # FixMe Help for item header is not descriptive
        item_header: str = typer.Option('title', help="Item header")):
    additional_features = {}
    next_item_split.create_conditional_index_using_extractor(data_file_path,
                                                             session_index_path,
                                                             output_dir_path / 'validation.idx',
                                                             item_header,
                                                             minimum_session_length,
                                                             delimiter,
                                                             additional_features,
                                                             next_item_split.get_position_with_offset_one)

    next_item_split.create_conditional_index_using_extractor(data_file_path, session_index_path,
                                                             output_dir_path / 'testing.idx',
                                                             item_header,
                                                             minimum_session_length,
                                                             delimiter,
                                                             additional_features,
                                                             next_item_split.get_position_with_offset_two)


@app.command()
def ratios(
        data_file_path: Path = typer.Argument(..., help="Data file in csv format"),
        session_index_path: Path = typer.Argument(..., help="Path to session index for the data file"),
        output_dir_path: Path = typer.Argument(..., help="path that the splits should be written to"),
        train_ratio: float = typer.Argument(0.9, help="a list of splits, e.g. train;0.9 valid;0.05 test;0.05"),
        validation_ratio: float = typer.Argument(0.05, help="a list of splits, e.g. train;0.9 valid;0.05 test;0.05"),
        testing_ratio: float = typer.Argument(0.05, help="a list of splits, e.g. train;0.9 valid;0.05 test;0.05"),
        seed: int = typer.Option(123456, help="Seed for random sampling of splits")):

    output_dir_path.mkdir(parents=True, exist_ok=True)
    assert train_ratio+validation_ratio+testing_ratio == 1
    splits = {"train": train_ratio, "valid": validation_ratio, "test": testing_ratio}
    ratio_split.run(data_file_path, session_index_path, output_dir_path, splits, seed)
