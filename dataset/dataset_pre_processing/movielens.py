import os
import functools
import pandas as pd
from pathlib import Path

from dataset.dataset_pre_processing.utils import build_vocabularies, read_csv, download_dataset, unzip_file
from dataset.dataset_index_splits.conditional_split import _get_position_with_offset, \
    create_conditional_index_using_extractor
from dataset.app.index_command import index_csv

DOWNLOAD_URL_MAP = {
    'ml-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-20m': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
}


def download_and_unzip_movielens_data(dataset: str, output_dir: Path, min_seq_length: int) -> (Path, Path):
    url = DOWNLOAD_URL_MAP[dataset]
    dataset_dir = output_dir / f'{dataset}_{min_seq_length}'
    download_dir = output_dir / dataset

    downloaded_file = download_dataset(url, download_dir)

    extract_dir = download_dir / 'raw'
    if not os.path.exists(extract_dir):
        extract_dir.mkdir(parents=True, exist_ok=True)

    unzip_file(downloaded_file, extract_dir, delete=False)
    return dataset_dir, extract_dir


def preprocess_movielens_data(dataset_dir: Path,
                              output_dir: Path,
                              name: str,
                              delimiter: str = '\t') -> Path:
    """
    Convert raw movielens data to csv files and create vocabularies
    :param delimiter:
    :param output_dir:
    :param dataset_dir:
    :param name:
    :return: the path to the main file
    """
    print("Convert to csv...")

    if name == "ml-1m":
        file_type = ".dat"
        header = None
        sep = "::"
    else:
        file_type = ".csv"
        header = 0
        sep = ","

    # read and merge data
    ratings_df = read_csv(dataset_dir, "ratings", file_type, sep, header)
    movies_df = read_csv(dataset_dir, "movies", file_type, sep, header)

    # only the ml-1m dataset has got a user info file …
    users_df = None

    if name == "ml-1m":
        ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']
        movies_df.columns = ['movieId', 'title', 'genres']
        users_df = read_csv(dataset_dir, "users", file_type, sep, header)
        users_df.columns = ['userId', 'gender', 'age', 'occupation', 'zip']
        ratings_df = pd.merge(ratings_df, users_df)

    elif name == "ml-20":
        links_df = read_csv(dataset_dir, "links", file_type, sep, header)
        ratings_df = pd.merge(ratings_df, links_df)

    merged_df = pd.merge(ratings_df, movies_df).sort_values(by=["userId", "timestamp"])
    # remove unused movie id column
    del merged_df['movieId']

    os.makedirs(output_dir, exist_ok=True)

    main_file = output_dir / f"{name}.csv"
    merged_df.to_csv(main_file, sep=delimiter, index=False)

    # build vocabularies
    # FIXME: the vocab should be build from the train set and not on the complete dataset
    build_vocabularies(movies_df, output_dir, "title")
    build_vocabularies(movies_df, output_dir, "genres", split="|")
    # for the ml-1m also export the vocabularies for the attributes
    if users_df is not None:
        build_vocabularies(users_df, output_dir, "gender")
        build_vocabularies(users_df, output_dir, "age")
        build_vocabularies(users_df, output_dir, "occupation")
        build_vocabularies(users_df, output_dir, "zip")

    return main_file


def split_movielens_dataset(dataset_dir: Path,
                            main_file: Path,
                            session_key: str = 'userId',
                            delimiter: str = '\t',
                            item_header: str = 'title',
                            min_seq_length: int = 4
                            ):
    # we use leave one out evaluation: the last watched movie for each users is in the test set, the second last is in
    # the valid test and the rest in the train set
    # we generate the session position index for validation and test (for train the validation index can be used the
    # validation index with the configuration that the target is not exposed to the model)
    index_file = dataset_dir / f'{main_file.stem}.idx'

    index_csv(main_file, index_file, [session_key], delimiter)

    additional_features = {}

    create_conditional_index_using_extractor(main_file, index_file, dataset_dir / 'valid.loo.idx', item_header,
                                             min_seq_length, delimiter, additional_features,
                                             functools.partial(_get_position_with_offset, offset=2))
    create_conditional_index_using_extractor(main_file, index_file, dataset_dir / 'test.loo.idx', item_header,
                                             min_seq_length, delimiter, additional_features,
                                             functools.partial(_get_position_with_offset, offset=1))
