"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DEPRECATED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import math
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Text, List
from tqdm import tqdm
from numpy.random._generator import default_rng

from data.base.reader import CsvDatasetIndex, CsvDatasetReader


def perform_split(split_ratios: Dict[Text, float], sample_indices: List[int]) -> Dict[Text, List[int]]:
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


def write_split(reader: CsvDatasetReader, output_dir_path: Path, header: Text, split_name: Text, sample_indices: List[int]):
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


def run(data_file_path: Path,
        match_index_path: Path,
        output_dir_path: Path,
        split_ratios: Dict[Text, float],
        seed: int):

    session_index = CsvDatasetIndex(match_index_path)
    reader = CsvDatasetReader(data_file_path, session_index)

    num_samples = len(session_index)
    sample_indices = list(range(num_samples))

    rng = default_rng(seed)
    rng.shuffle(sample_indices)

    splits = perform_split(split_ratios, sample_indices)

    header = get_header(data_file_path)

    for split_name, sample_indices in splits.items():
        write_split(reader, output_dir_path, header, split_name, sample_indices)


def extract_splits(split_desc: List[str]) -> Dict[str, float]:
    return {desc.split(";")[0]: float(desc.split(";")[1]) for desc in split_desc}


if __name__ == "__main__":
    parser = ArgumentParser('split a csv file (with its session index file) in train, valid and test files')

    parser.add_argument("data_file", help="data file in csv format", type=str)
    parser.add_argument("session_index_file", help="session index for the data file.", type=str)
    parser.add_argument("output_dir", help="output directory", type=str)
    parser.add_argument("splits", nargs="+", help="a list of splits, e.g. train;0.9 valid;0.05 test;0.05")
    parser.add_argument("--seed", help="a seed", type=int, default=123456)

    args = parser.parse_args()

    data_file_path = Path(args.data_file)
    session_index_path = Path(args.session_index_file)
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    splits = extract_splits(args.splits)
    seed = args.seed

    run(data_file_path, session_index_path, output_dir_path, splits, seed)
