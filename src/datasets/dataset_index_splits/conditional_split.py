import functools
from operator import itemgetter
from pathlib import Path
from typing import Dict, Any, Iterable, Callable, Optional
from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets.index_builder import SequencePositionIndexBuilder
from data.datasets.sequence import ItemSequenceDataset, ItemSessionParser, PlainSequenceDataset
from data.utils import create_indexed_header, read_csv_header
from datasets.data_structures.dataset_metadata import DatasetMetadata


def all_remaining_positions(session: Dict[str, Any]
                            ) -> Iterable[int]:
    return range(1, len(session[ITEM_SEQ_ENTRY_NAME]) - 2)


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


def create_conditional_index(dataset_metadata: DatasetMetadata,
                             output_file_path: Path,
                             target_feature: Optional[str]) -> None:
    """
    FixMe I need some documentation
    :param dataset_metadata: Data Set Metadata
    :param output_file_path: path to the output file
    :param target_feature: the target column name to build the targets against,
    (default all next subsequences will be considered); the target must be a boolean feature
    :return:
    """
    target_positions_extractor = _build_target_position_extractor(target_feature)
    additional_features = {}
    if target_feature is not None:
        additional_features[target_feature] = {'type': 'bool', 'sequence': True}

    create_conditional_index_using_extractor(dataset_metadata=dataset_metadata,
                                             output_file_path=output_file_path,
                                             target_positions_extractor=target_positions_extractor,
                                             additional_features=additional_features)


def create_conditional_index_using_extractor(dataset_metadata: DatasetMetadata,
                                             output_file_path: Path,
                                             target_positions_extractor: Callable[[Dict[str, Any]], Iterable[int]],
                                             additional_features: Optional[Dict[str, Any]] = None) -> None:
    """
    Create a new conditional Index from an already existing one via target position extractor.
    :param dataset_metadata: Data Set Metadata
    :param output_file_path: file that the index is stored to
    :param additional_features: FixMe I need a description
    :param target_positions_extractor: FixMe I need a description
    :return: None, Side effect: new index file is stored at output_file_path
    """
    if additional_features is None:
        additional_features = {}

    session_parser = ItemSessionParser(
        create_indexed_header(read_csv_header(dataset_metadata.data_file_path, dataset_metadata.delimiter)),
        item_header_name=dataset_metadata.item_header_name,
        delimiter=dataset_metadata.delimiter,
        additional_features=additional_features
    )

    session_index = CsvDatasetIndex(dataset_metadata.session_index_path)
    reader = CsvDatasetReader(dataset_metadata.data_file_path, session_index)

    plain_dataset = PlainSequenceDataset(reader, session_parser)
    dataset = ItemSequenceDataset(plain_dataset)

    builder = SequencePositionIndexBuilder(target_positions_extractor=target_positions_extractor)
    builder.build(dataset, output_file_path)


def run_loo_split(dataset_metadata: DatasetMetadata,
                  output_dir_path: Path):
    """
    Creates a next item split, i.e., From every session with length k use sequence[0:k-2] for training,
    sequence[-2] for validation and sequence[-1] for testing.

    :param dataset_metadata: Data Set Metadata
    :param output_dir_path: output directory where the index files for the splits are written to
    :param minimum_session_length: Minimum length that sessions need to be in order to be included
    :return:
    """
    # Create validation index with target item n-1
    create_conditional_index_using_extractor(dataset_metadata=dataset_metadata, output_file_path=output_dir_path / (
                dataset_metadata.file_prefix + ".validation.loo.idx"),
                                             target_positions_extractor=get_position_with_offset_two)
    # Create testing index with target item n
    create_conditional_index_using_extractor(dataset_metadata=dataset_metadata, output_file_path=output_dir_path / (
                dataset_metadata.file_prefix + ".test.loo.idx"),
                                             target_positions_extractor=get_position_with_offset_one)
