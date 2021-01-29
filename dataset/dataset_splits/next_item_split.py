import typer
from pathlib import Path
from typing import Dict, Any, Iterable, Callable
from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets.index_builder import SessionPositionIndexBuilder
from data.datasets.session import ItemSessionDataset, ItemSessionParser, PlainSessionDataset
from data.utils import create_indexed_header, read_csv_header


def _get_position_with_offset(session: Dict[str, Any], offset: int) -> Iterable[int]:
    sequence = session[ITEM_SEQ_ENTRY_NAME]
    return [len(sequence) - offset]


def get_position_with_offset_one(session: Dict[str, Any]) -> Iterable[int]:
    """
    Helper method for testing index
    :param session: session that the target is to be extracted from
    :return: index within session where testing target is stored
    """
    return _get_position_with_offset(session, offset=1)


def get_position_with_offset_two(session: Dict[str, Any]) -> Iterable[int]:
    """
    Helper method for validation index
    :param session: session that the target is to be extracted from
    :return: index within session where validation target is stored
    """
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
    """
    Create a new conditional Index from an already existing one via target position extractor.
    :param data_file_path: data file belonging to original index (session_index_path)
    :param session_index_path: original index that conditional index should be extracted from
    :param output_file_path: file that the index is stored to
    :param item_header_name: Session ID key
    :param min_session_length: Minimum session length for sessions that are stored in conditional index
    :param delimiter: delimiter used in original data file
    :param additional_features: FixMe I need a description
    :param target_positions_extractor: FixMe I need a description
    :return: None, Side effect: new index file is stored at :param output_file_path:
    """
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
