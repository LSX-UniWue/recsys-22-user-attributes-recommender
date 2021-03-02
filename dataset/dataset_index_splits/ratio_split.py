import math

from pathlib import Path
from typing import Dict, Text, List
from tqdm import tqdm
from numpy.random._generator import default_rng
from data.base.reader import CsvDatasetIndex, CsvDatasetReader

from dataset.app import index_command
from dataset.dataset_index_splits.conditional_split import create_conditional_index


def run(data_file_path: Path,
        match_index_path: Path,
        output_dir_path: Path,
        session_key: List[str],
        split_ratios: Dict[Text, float],
        delimiter: str,
        item_header_name: str,
        minimum_session_length: int,
        seed: int):
    """
    ToDo
    :param data_file_path:
    :param match_index_path:
    :param output_dir_path:
    :param session_key: Session identifier name in data set header
    :param split_ratios:
    :param delimiter: Delimiter used in original data file
    :param minimum_session_length: Minimum length that sessions need to be in order to be included
    :param item_header_name: data set key that the item-ids are stored under
    :param seed:
    :return:
    """
    file_name: str = data_file_path.stem
    session_index = CsvDatasetIndex(match_index_path)
    reader = CsvDatasetReader(data_file_path, session_index)

    num_samples = len(session_index)
    sample_indices = list(range(num_samples))

    rng = default_rng(seed)
    rng.shuffle(sample_indices)

    splits = perform_ratio_split(split_ratios, sample_indices)

    header = get_header(data_file_path)

    for split_name, sample_indices in splits.items():
        write_split(reader, output_dir_path, header, file_name + "." + split_name, sample_indices)

    # Index newly written splits
    for split in tqdm(["train", "test", "validation"], desc="Index new splits"):
        file_prefix: str = file_name + "." + split
        data_file = output_dir_path.joinpath(file_prefix + ".csv")
        split_index_file = output_dir_path.joinpath(file_prefix + ".session.idx")
        next_item_index_file = output_dir_path.joinpath(file_prefix + ".nextitem.idx")
        index_command.index_csv(data_file_path=data_file, index_file_path=split_index_file,
                                session_key=session_key, delimiter=delimiter)
        create_conditional_index(data_file_path=data_file,
                                 session_index_path=split_index_file,
                                 output_file_path=next_item_index_file,
                                 item_header_name=item_header_name,
                                 min_session_length=minimum_session_length,
                                 delimiter=delimiter,
                                 target_feature=None)


def perform_ratio_split(split_ratios: Dict[Text, float], sample_indices: List[int]) -> Dict[Text, List[int]]:
    """
    ToDo Document me
    :param split_ratios:
    :param sample_indices:
    :return:
    """
    num_samples = len(sample_indices)
    remainder = sample_indices
    splits = dict()
    for split_name, ratio in split_ratios.items():
        remainder_length = len(remainder)
        num_samples_in_split = int(math.ceil(ratio * num_samples))

        # take only what is left for the last split to avoid errors
        num_samples_in_split = min(num_samples_in_split, remainder_length)

        samples = remainder[:num_samples_in_split]
        remainder = remainder[num_samples_in_split:]

        splits[split_name] = samples

    return splits


def write_split(reader: CsvDatasetReader, output_dir_path: Path, header: Text, split_name: Text,
                sample_indices: List[int]):
    """
    ToDo Document me
    :param reader:
    :param output_dir_path:
    :param header:
    :param split_name:
    :param sample_indices:
    :return:
    """
    output_file = output_dir_path / f"{split_name}.csv"
    with output_file.open("w") as file:
        file.write(header)
        file.write("\n")
        for idx in tqdm(sample_indices, desc=f"Generating {split_name}"):
            content = reader.get_session(idx)
            file.write(content.strip())
            file.write("\n")


def get_header(data_file_path: Path) -> Text:
    with data_file_path.open("r") as file:
        return file.readline().strip()


def extract_splits(split_desc: List[str]) -> Dict[str, float]:
    return {desc.split(";")[0]: float(desc.split(";")[1]) for desc in split_desc}
