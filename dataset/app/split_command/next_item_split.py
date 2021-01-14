import typer
from pathlib import Path
from typing import List, Dict, Any, Iterable, Callable
from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets.index_builder import SessionPositionIndexBuilder
from data.datasets.session import ItemSessionDataset, ItemSessionParser, PlainSessionDataset
from data.utils import create_indexed_header, read_csv_header
from dataset.app.split_command import app


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

    create_conditional_index_using_extractor(data_file_path,
                                             session_index_path,
                                             output_dir_path / 'validation.idx',
                                             item_header,
                                             minimum_session_length, delimiter,
                                             additional_features,
                                             _get_position_with_offset_one)

    create_conditional_index_using_extractor(data_file_path, session_index_path,
                                             output_dir_path / 'testing.idx',
                                             item_header,
                                             minimum_session_length,
                                             delimiter, additional_features,
                                             _get_position_with_offset_two)


def _get_position_with_offset(session: Dict[str, Any],
                              offset: int
                              ) -> Iterable[int]:
    sequence = session[ITEM_SEQ_ENTRY_NAME]
    return [len(sequence) - offset]


def _get_position_with_offset_one(session: Dict[str, Any]) -> Iterable[int]:
    return _get_position_with_offset(session, offset=1)


def _get_position_with_offset_two(session: Dict[str, Any]) -> Iterable[int]:
    return _get_position_with_offset(session, offset=2)


def create_conditional_index_using_extractor(data_file_path: Path,
                                             session_index_path: Path,
                                             output_file_path: Path,
                                             item_header_name: str,
                                             min_session_length: int,
                                             delimiter: str,
                                             additional_features: Dict[str, Any],
                                             target_positions_extractor: Callable[[Dict[str, Any]], Iterable[int]]
                                             ) -> None:
    session_parser = ItemSessionParser(
        create_indexed_header(read_csv_header(data_file_path, delimiter)),
        item_header_name,
        delimiter=delimiter,
        additional_features=additional_features
    )

    session_index = CsvDatasetIndex(session_index_path)
    reader = CsvDatasetReader(data_file_path, session_index)

    plain_dataset = PlainSessionDataset(reader, session_parser)
    dataset = ItemSessionDataset(plain_dataset)

    builder = SessionPositionIndexBuilder(min_session_length=min_session_length,
                                          target_positions_extractor=target_positions_extractor)
    builder.build(dataset, output_file_path)
