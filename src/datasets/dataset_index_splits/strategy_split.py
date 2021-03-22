from pathlib import Path
from typing import Dict, Text, List, Optional, Set, Callable, Any
from tqdm import tqdm
from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets.sequence import ItemSessionParser
from data.utils import create_indexed_header
from data.datasets import ITEM_SEQ_ENTRY_NAME

from datasets.app import index_command
from datasets.dataset_index_splits.conditional_split import create_conditional_index, read_csv_header
from datasets.data_structures.DatasetMetadata import DatasetMetadata
from datasets.data_structures.TrainValidationTestSplitsIndices import TrainValidationTestSplitIndices
from datasets.data_structures.SplitNames import SplitNames
from datasets.data_structures.SplitStrategy import SplitStrategy


def run_strategy_split(dataset_metadata: DatasetMetadata,
                       output_dir_path: Path,
                       split_strategy: SplitStrategy,
                       minimum_session_length: int):
    """
    :param dataset_metadata: Data set metadata
    :param output_dir_path: output directory where the data and index files for the splits are written to
    :param split_strategy: Strategy that dictates the training, test, validation split
    :param minimum_session_length: Minimum length that sessions need to be in order to be included
    :return: None, Side effects: CSV Files for splits are written
    """
    output_dir_path.mkdir(parents=True, exist_ok=True)
    splits = split_strategy.split(dataset_metadata)
    write_all_splits(dataset_metadata=dataset_metadata, splits=splits, output_dir_path=output_dir_path)
    index_splits(dataset_metadata=dataset_metadata, output_dir_path=output_dir_path,
                 minimum_session_length=minimum_session_length)


def write_all_splits(dataset_metadata: DatasetMetadata, splits: TrainValidationTestSplitIndices, output_dir_path: Path):
    """

    :param dataset_metadata:
    :param splits:
    :param output_dir_path:
    :return:
    """
    # Read Splits via Session index
    session_index = CsvDatasetIndex(dataset_metadata.session_index_path)
    header = get_header(dataset_metadata.data_file_path)
    parser: ItemSessionParser = ItemSessionParser(
        indexed_headers=create_indexed_header(
            read_csv_header(data_file_path=dataset_metadata.data_file_path,
                            delimiter=dataset_metadata.delimiter)
        ),
        item_header_name=dataset_metadata.item_header_name,
        delimiter=dataset_metadata.delimiter
    )
    reader = CsvDatasetReader(dataset_metadata.data_file_path, session_index)
    # Get all unique training items during writing of the training split
    training_items: List[str] = write_training_split(reader=reader,
                                                     output_dir_path=output_dir_path,
                                                     header=header,
                                                     output_file_name=f"{dataset_metadata.file_prefix}.{SplitNames.train}",
                                                     sample_indices=splits.get(SplitNames.train),
                                                     parser=parser)
    unique_items_set: Set[str] = get_unique_items(session_index=session_index, reader=reader, parser=parser)

    non_training_items: Set[str] = unique_items_set - set(training_items)
    # Write Validation and Test split
    for split_name in [SplitNames.validation, SplitNames.test]:
        write_split(reader=reader,
                    output_dir_path=output_dir_path,
                    header=header,
                    output_file_name=dataset_metadata.file_prefix + "." + str(split_name),
                    sample_indices=splits.get(split_name),
                    non_training_items=non_training_items, parser=parser)


def write_training_split(reader: CsvDatasetReader, output_dir_path: Path, header: Text, output_file_name: Text,
                         sample_indices: List[int], parser: ItemSessionParser):
    """
    ToDo Document me
    :param reader:
    :param output_dir_path:
    :param header:
    :param output_file_name:
    :param sample_indices:
    :param parser
    :return:
    """

    output_file = output_dir_path / f"{output_file_name}.csv"
    training_items: List[str] = []
    with output_file.open("w") as file:
        file.write(header)
        file.write("\n")
        for idx in tqdm(sample_indices, desc=f"Generating {output_file_name}"):
            content = reader.get_sequence(idx)
            # drop sessions that contain items that do not exist in the train set
            # Get Items of a session
            session_items: List[str] = parser.parse(content).get(ITEM_SEQ_ENTRY_NAME)
            training_items += session_items
            file.write(content.strip())
            file.write("\n")
    return training_items


def write_split(reader: CsvDatasetReader, output_dir_path: Path, header: Text, output_file_name: Text,
                sample_indices: List[int], non_training_items: Optional[Set[str]], parser: ItemSessionParser,
                verbose=False):
    """
    ToDo Document me
    :param reader:
    :param output_dir_path:
    :param header:
    :param output_file_name:
    :param sample_indices:
    :param non_training_items
    :param parser
    :param verbose prints dropped sessions if set to True
    :return:
    """
    output_file = output_dir_path / f"{output_file_name}.csv"
    with output_file.open("w") as file:
        file.write(header)
        file.write("\n")
        for idx in tqdm(sample_indices, desc=f"Generating {output_file_name}"):
            content = reader.get_sequence(idx)
            # drop sessions that contain items that do not exist in the train set
            # Get Items of a session
            session_items: List[str] = parser.parse(content).get(ITEM_SEQ_ENTRY_NAME)
            # Check for overlap between non training items and session items
            if any(set(session_items).intersection(non_training_items)):
                if verbose:
                    print(f"{output_file_name} dropped Session {idx}\n"
                          f"Session items:\n"
                          f"{session_items}\n"
                          f"Overlap:\n"
                          f"{set(session_items).intersection(non_training_items)}\n")
                continue
            file.write(content.strip())
            file.write("\n")


def index_splits(dataset_metadata: DatasetMetadata, output_dir_path: Path, minimum_session_length: int):
    # Index newly written splits
    for split in tqdm(SplitNames, desc="Index new splits"):
        file_prefix: str = dataset_metadata.file_prefix + "." + str(split)
        data_file = output_dir_path.joinpath(file_prefix + ".csv")
        split_index_file = output_dir_path.joinpath(file_prefix + ".session.idx")
        next_item_index_file = output_dir_path.joinpath(file_prefix + ".nextitem.idx")
        index_command.index_csv(data_file_path=data_file, index_file_path=split_index_file,
                                session_key=dataset_metadata.session_key, delimiter=dataset_metadata.delimiter)
        create_conditional_index(dataset_metadata=dataset_metadata,
                                 output_file_path=next_item_index_file,
                                 min_session_length=minimum_session_length,
                                 target_feature=None)


def get_header(data_file_path: Path) -> Text:
    with data_file_path.open("r") as file:
        return file.readline().strip()


def get_unique_items(session_index: CsvDatasetIndex, reader: CsvDatasetReader, parser: ItemSessionParser):
    result: Set[str] = set()
    for idx in range(session_index.num_sequences()):
        content = reader.get_sequence(idx)
        for item in parser.parse(content).get(ITEM_SEQ_ENTRY_NAME):
            result.add(item)
    return result


def extract_splits(split_desc: List[str]) -> Dict[str, float]:
    return {desc.split(";")[0]: float(desc.split(";")[1]) for desc in split_desc}
