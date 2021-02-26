import typer
from pathlib import Path
from typing import List
from dataset.dataset_index_splits import conditional_split, ratio_split

app = typer.Typer()


@app.command()
def next_item(
        data_file_path: Path = typer.Argument(..., help="Data file in csv format"),
        session_index_path: Path = typer.Argument(..., help="Path to session index for the data file"),
        output_dir_path: Path = typer.Argument(..., help="path that the splits should be written to"),
        minimum_session_length: int = typer.Option(4, help="Minimum length of sessions that are to be included"),
        delimiter: str = typer.Option('\t', help="Delimiter used in data file"),
        item_header: str = typer.Option('title', help="Dataset Key that the Item-IDs are stored under")):
    """
    Creates a next item split, i.e., From every session with length k use sequence[0:k-2] for training,
    sequence[-2] for validation and sequence[-1] for testing.

    :param data_file_path: data file that the split should be created for
    :param session_index_path: session index belonging to the data file
    :param output_dir_path: output directory where the index files for the splits are written to
    :param minimum_session_length: Minimum length that sessions need to be in order to be included
    :param delimiter: delimiter used in data file
    :param item_header: data set key that the item-ids are stored under
    :return: None, Side effect: Test and Validation indices are written
    """
    additional_features = {}

    # Create validation index with target item n-1
    conditional_split.create_conditional_index_using_extractor(data_file_path,
                                                               session_index_path,
                                                               output_dir_path / 'valid.idx',
                                                               item_header,
                                                               minimum_session_length,
                                                               delimiter,
                                                               additional_features,
                                                               conditional_split.get_position_with_offset_one)
    # Create testing index with target item n
    conditional_split.create_conditional_index_using_extractor(data_file_path, session_index_path,
                                                               output_dir_path / 'test.idx',
                                                               item_header,
                                                               minimum_session_length,
                                                               delimiter,
                                                               additional_features,
                                                               conditional_split.get_position_with_offset_two)


@app.command()
def ratios(
        data_file_path: Path = typer.Argument(..., help="Data file in csv format"),
        session_index_path: Path = typer.Argument(..., help="Path to session index for the data file"),
        output_dir_path: Path = typer.Argument(..., help="path that the splits should be written to"),
        session_key: List[str] = typer.Argument(..., help="session key"),
        item_header_name: str = typer.Argument(..., help="name of the column that contains the item id"),
        minimum_session_length: int = typer.Option(2, help="the minimum acceptable session length"),
        train_ratio: float = typer.Argument(0.9, help="a list of splits, e.g. train;0.9 valid;0.05 test;0.05"),
        validation_ratio: float = typer.Argument(0.05, help="a list of splits, e.g. train;0.9 valid;0.05 test;0.05"),
        testing_ratio: float = typer.Argument(0.05, help="a list of splits, e.g. train;0.9 valid;0.05 test;0.05"),
        delimiter: str = typer.Option('\t', help="Delimiter used in data file"),
        seed: int = typer.Option(123456, help="Seed for random sampling of splits")):
    """
    Creates a data set split based on ratios where a percentage of the sessions are used for training, validation, and
    testing.
    :param data_file_path: data file that the split should be created for
    :param session_index_path: session index belonging to the data file
    :param output_dir_path: output directory where the data and index files for the splits are written to
    :param session_key: Session key used to uniquely identify sessions
    :param minimum_session_length: Minimum length that sessions need to be in order to be included
    :param item_header_name: data set key that the item-ids are stored under
    :param train_ratio: share of session used for training
    :param validation_ratio: share of session used for validation
    :param testing_ratio: share of session used for testing
    :param delimiter: delimiter used in data file
    :param seed: Seed for random sampling
    :return: None, Side effects: CSV Files for splits are written
     """
    output_dir_path.mkdir(parents=True, exist_ok=True)
    assert train_ratio + validation_ratio + testing_ratio == 1
    splits = {"train": train_ratio, "valid": validation_ratio, "test": testing_ratio}
    ratio_split.run(data_file_path=data_file_path,
                    match_index_path=session_index_path,
                    output_dir_path=output_dir_path,
                    session_key=session_key,
                    split_ratios=splits,
                    delimiter=delimiter,
                    seed=seed,
                    item_header_name=item_header_name,
                    minimum_session_length=minimum_session_length)


# Todo Find better name
@app.command()
def create_conditional_index(
        data_file_path: Path = typer.Argument(..., exists=True, help="path to the input file in CSV format"),
        session_index_path: Path = typer.Argument(..., exists=True, help="path to the session index file"),
        output_file_path: Path = typer.Argument(..., help="path to the output file"),
        item_header_name: str = typer.Argument(..., help="name of the column that contains the item id"),
        min_session_length: int = typer.Option(2, help="the minimum acceptable session length"),
        delimiter: str = typer.Option("\t", help="the delimiter used in the CSV file."),
        target_feature: str = typer.Option(None, help="the target column name to build the targets against"
                                                      "(default all next subsequences will be considered);"
                                                      "the target must be a boolean feature")
) -> None:
    """
    FixMe I need some documentation
    :param data_file_path:
    :param session_index_path:
    :param output_file_path:
    :param item_header_name:
    :param min_session_length:
    :param delimiter:
    :param target_feature:
    :return:
    """
    # Builds
    target_positions_extractor = conditional_split._build_target_position_extractor(target_feature)
    additional_features = {}
    if target_feature is not None:
        additional_features[target_feature] = {'type': 'bool', 'sequence': True}

    conditional_split.create_conditional_index_using_extractor(data_file_path, session_index_path, output_file_path,
                                                               item_header_name,
                                                               min_session_length, delimiter, additional_features,
                                                               target_positions_extractor)
