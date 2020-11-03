import functools
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import Dict, Any, Iterable, Callable, Optional

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets.nextitem import NextItemIndexBuilder
from data.datasets.session import ItemSessionDataset, ItemSessionParser
from data.utils import create_indexed_header, read_csv_header


def filter_by_sequence_feature(session: Dict[str, Any],
                               feature_key: str
                               ) -> Iterable[int]:
    feature_values = session[feature_key]
    targets = list(filter(itemgetter(1), enumerate(feature_values)))
    target_idxs = list(map(itemgetter(0), targets))
    if 0 in target_idxs:
        target_idxs.remove(0)  # XXX: quick hack to remove the first sequence that
    return target_idxs


def _build_target_position_extractor(target_feature: str
                                     ) -> Optional[Callable[[Dict[str, Any]], Iterable[int]]]:
    if target_feature is None:
        return None

    return functools.partial(filter_by_sequence_feature, feature_key=target_feature)


def run(data_file_path: Path,
        session_index_path: Path,
        output_file_path: Path,
        item_header_name: str,
        min_session_length: int = 2,
        delimiter: str = "\t",
        target_feature: str = None):

    target_positions_extractor = _build_target_position_extractor(target_feature)

    session_index = CsvDatasetIndex(session_index_path)

    reader = CsvDatasetReader(data_file_path, session_index)

    additional_features = {}
    if target_feature is not None:
        additional_features[target_feature] = {'type': 'bool', 'sequence': True}

    session_parser = ItemSessionParser(
        create_indexed_header(read_csv_header(data_file_path, delimiter)),
        item_header_name,
        delimiter=delimiter,
        additional_features=additional_features
    )
    dataset = ItemSessionDataset(reader, session_parser)
    builder = NextItemIndexBuilder(min_session_length=min_session_length,
                                   target_positions_extractor=target_positions_extractor)
    builder.build(dataset, output_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_file", help="path to the csv data file", type=str)
    parser.add_argument("session_index_file", help="path to the session index file", type=str)
    parser.add_argument("output_file", help="path where the index is written to", type=str)
    parser.add_argument("item_header_name", help="name of the item column (header)", type=str)
    parser.add_argument("--min_session_length", help="minimum session length (Default: 2)", default=2, type=int)
    parser.add_argument("--delimiter", help="delimiter used in the csv data file (Default: \\t).", default="\t",
                        type=str)
    parser.add_argument('--target_feature', help='the target column name to build the targets against (default all next'
                                                 'subsequences will be considered); the target must be a boolean'
                                                 'feature',
                        default=None, type=str)

    args = parser.parse_args()

    data_file = Path(args.data_file)
    session_index_file = Path(args.session_index_file)
    output_file = Path(args.output_file)
    item_header_name = args.item_header_name

    min_session_length = args.min_session_length
    delimiter = args.delimiter

    run(data_file, session_index_file, output_file, item_header_name, min_session_length, delimiter,
        args.target_feature)
