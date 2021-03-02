import functools
from operator import itemgetter
from pathlib import Path
from typing import Dict, Any, Iterable, Callable, Optional
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
    :return: None, Side effect: new index file is stored at output_file_path
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


def filter_by_sequence_feature(session: Dict[str, Any],
                               feature_key: str
                               ) -> Iterable[int]:
    feature_values = session[feature_key]
    targets = list(filter(itemgetter(1), enumerate(feature_values)))
    target_idxs = list(map(itemgetter(0), targets))
    if 0 in target_idxs:
        target_idxs.remove(0)  # XXX: quick hack to remove the first position that can not be predicted by the models
    return target_idxs


def _build_target_position_extractor(target_feature: str
                                     ) -> Optional[Callable[[Dict[str, Any]], Iterable[int]]]:
    if target_feature is None:
        return None

    return functools.partial(filter_by_sequence_feature, feature_key=target_feature)


def create_conditional_index(
        data_file_path: Path,
        session_index_path: Path,
        output_file_path: Path,
        item_header_name: str,
        min_session_length: int,
        delimiter: str,
        target_feature: Optional[str]
) -> None:
    """
    FixMe I need some documentation
    :param data_file_path: path to the input file in CSV format
    :param session_index_path: path to the session index file
    :param output_file_path: path to the output file
    :param item_header_name: name of the column that contains the item id
    :param min_session_length: the minimum acceptable session length
    :param delimiter: the delimiter used in the CSV file.
    :param target_feature: the target column name to build the targets against,
    (default all next subsequences will be considered); the target must be a boolean feature
    :return:
    """
    target_positions_extractor = _build_target_position_extractor(target_feature)
    additional_features = {}
    if target_feature is not None:
        additional_features[target_feature] = {'type': 'bool', 'sequence': True}

    create_conditional_index_using_extractor(data_file_path, session_index_path, output_file_path,
                                             item_header_name,
                                             min_session_length, delimiter, additional_features,
                                             target_positions_extractor)
