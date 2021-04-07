from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Callable, Any

from datasets.data_structures.dataset_metadata import DatasetMetadata
from datasets.app import index_command
from datasets.dataset_index_splits.conditional_split import all_remaining_positions, \
    create_conditional_index_using_extractor, run_loo_split
from datasets.dataset_index_splits import strategy_split
from datasets.vocabulary.create_vocabulary import create_token_vocabulary
from datasets.popularity.build_popularity import build as build_popularity
from datasets.dataset_index_splits import split_strategies_factory
from datasets.data_structures.split_strategy import SplitStrategy


def generic_process_dataset(dataset_metadata: DatasetMetadata,
                            min_seq_length: int
                            ) -> None:
    """
    Handles pre-processing, splitting, storing and indexing of a given session data sets.

    :param dataset_metadata: Data set metadata
    :param min_seq_length: Minimum length of a session in order to be included in the next item split
    :return: Todo
    """
    # Index pre-processed data
    print("Indexing processed data...")
    index_command.index_csv(data_file_path=dataset_metadata.data_file_path,
                            index_file_path=dataset_metadata.session_index_path,
                            session_key=dataset_metadata.session_key,
                            delimiter=dataset_metadata.delimiter)

    print("Create ratios split...")
    ratio_split_strategy: SplitStrategy = split_strategies_factory.get_ratio_strategy(train_ratio=0.9,
                                                                                      validation_ratio=0.05,
                                                                                      test_ratio=0.05,
                                                                                      seed=123456)
    generic_strategy_split(dataset_metadata=dataset_metadata,
                           split_strategy=ratio_split_strategy,
                           minimum_session_length=min_seq_length)

    print("Create Leave One Out split...")
    generic_leave_one_out_split(dataset_metadata=dataset_metadata,
                                minimum_session_length=min_seq_length)


def generic_strategy_split(
        dataset_metadata: DatasetMetadata,
        split_strategy: SplitStrategy,
        minimum_session_length: int
):
    """
    Generating ratio_split, as well as corresponding vocabulary and popularity files for a specified data file and index
    :param dataset_metadata: Data set metadata
    :param minimum_session_length: the minimum acceptable session length
    :param split_strategy: Strategy that dictates the training, test, validation split
    :return: Todo
    """
    split_output_dir_path: Path = dataset_metadata.dataset_base_dir.joinpath(str(split_strategy))

    strategy_split.run_strategy_split(dataset_metadata=dataset_metadata,
                                      output_dir_path=split_output_dir_path,
                                      split_strategy=split_strategy,
                                      minimum_session_length=minimum_session_length)
    train_file_path: Path = split_output_dir_path / f"{dataset_metadata.file_prefix}.train.csv"
    train_session_index_file_path: Path = split_output_dir_path / f"{dataset_metadata.file_prefix}.train.session.idx"
    print("Build vocabulary and popularity for ratio split...")
    generic_create_vocabularies_and_popularities(dataset_metadata=dataset_metadata,
                                                 processed_data_file_path=train_file_path,
                                                 index_path=train_session_index_file_path,
                                                 output_dir_path=split_output_dir_path,
                                                 minimum_session_length=minimum_session_length,
                                                 delimiter=dataset_metadata.delimiter)


def generic_leave_one_out_split(
        dataset_metadata: DatasetMetadata,
        minimum_session_length: int
):
    """
    Creates a next item split, i.e., From every session with length k use sequence[0:k-2] for training,
    sequence[-2] for validation and sequence[-1] for testing. As well as

    :param dataset_metadata: Data set metadata
    :param minimum_session_length: Minimum length that sessions need to be in order to be included
    :return: None, Side effect: Test and Validation indices are written
    """
    loo_output_dir_path: Path = dataset_metadata.dataset_base_dir / "loo"

    print("Create next item index...")
    # Create session index for training with all session having the form session[:k-2]
    training_session_index_file: Path = loo_output_dir_path / f"{dataset_metadata.file_prefix}.train.nextitem.idx"
    create_conditional_index_using_extractor(dataset_metadata=dataset_metadata,
                                             output_file_path=training_session_index_file,
                                             min_session_length=minimum_session_length,
                                             target_positions_extractor=all_remaining_positions)

    print("Create leave one out index...")
    run_loo_split(dataset_metadata=dataset_metadata,
                  output_dir_path=loo_output_dir_path,
                  minimum_session_length=minimum_session_length)

    print("Write vocabularies and popularities...")
    generic_create_vocabularies_and_popularities(dataset_metadata=dataset_metadata,
                                                 processed_data_file_path=dataset_metadata.data_file_path,
                                                 index_path=dataset_metadata.session_index_path,
                                                 output_dir_path=loo_output_dir_path,
                                                 minimum_session_length=minimum_session_length,
                                                 delimiter=dataset_metadata.delimiter)


def generic_create_vocabularies_and_popularities(
        dataset_metadata: DatasetMetadata,
        processed_data_file_path: Path,
        index_path: Path,
        output_dir_path: Path,
        minimum_session_length: int,
        delimiter: str,
        strategy_function: Optional[Callable[[List[Any]], List[Any]]] = None
):
    """
    Writes vocabularies and popularity for the specified columns in the index and data file.

    :param dataset_metadata: Data Set metadata
    :param processed_data_file_path: data file that the split should be created for
    :param index_path: session index belonging to the data file
    :param output_dir_path: output directory where the index files for the splits are written to
    :param minimum_session_length: Minimum length that sessions need to be in order to be included
    :param delimiter: delimiter used in data file
    :param strategy_function: function selecting which items of a session are used in the vocabulary and popularity
    :return: None, Side effect: Popularities and vocabularies are written
    """
    file_prefix: str = dataset_metadata.file_prefix
    for column_name in tqdm(dataset_metadata.stats_columns,
                            desc="Create Vocabulary and Popularity for every Stats Column"):
        vocabulary_output_file_path: Path = output_dir_path / f"{file_prefix}.vocabulary.{column_name}.txt"
        popularity_output_file_path: Path = output_dir_path / f"{file_prefix}.popularity.{column_name}.txt"

        create_token_vocabulary(item_header_name=column_name,
                                data_file_path=processed_data_file_path,
                                session_index_path=index_path,
                                vocabulary_output_file_path=vocabulary_output_file_path,
                                custom_tokens=dataset_metadata.custom_tokens,
                                delimiter=delimiter,
                                strategy_function=strategy_function)
        build_popularity(data_file_path=processed_data_file_path,
                         session_index_path=index_path,
                         vocabulary_file_path=vocabulary_output_file_path,
                         output_file_path=popularity_output_file_path,
                         item_header_name=column_name,
                         min_session_length=minimum_session_length,
                         delimiter=delimiter,
                         strategy_function=strategy_function)
