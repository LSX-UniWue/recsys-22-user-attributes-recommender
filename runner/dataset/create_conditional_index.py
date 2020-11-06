import functools
from operator import itemgetter
from pathlib import Path
from typing import Dict, Any, Iterable, Callable, Optional

import typer

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets.nextitem import NextItemIndexBuilder
from data.datasets.session import ItemSessionDataset, ItemSessionParser
from data.utils import create_indexed_header, read_csv_header

app = typer.Typer()


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


@app.command()
def run(data_file_path: Path = typer.Argument(..., help="path to the input file in CSV format"),
        session_index_path: Path = typer.Argument(..., help="path to the session index file"),
        output_file_path: Path = typer.Argument(..., help="path to the output file"),
        item_header_name: str = typer.Argument(..., help="name of the column that contains the item id"),
        min_session_length: int = typer.Option(2, help="the minimum acceptable session length"),
        delimiter: str = typer.Option("\t", help="the delimiter used in the CSV file."),
        target_feature: str = typer.Option(None, help="the target column name to build the targets against"
                                                      "(default all next subsequences will be considered);"
                                                      "the target must be a boolean feature")
        ) -> None:

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
    app()