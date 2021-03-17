import os
import functools
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, Any

from data.datasets import ITEM_SEQ_ENTRY_NAME
from datasets.dataset_pre_processing.utils import build_vocabularies, read_csv, download_dataset, unzip_file
from datasets.dataset_index_splits.conditional_split import _get_position_with_offset, \
    create_conditional_index_using_extractor
from datasets.app.index_command import index_csv

RATING_USER_COLUMN_NAME = 'userId'
RATING_MOVIE_COLUMN_NAME = 'movieId'
RATING_TIMESTAMP_COLUMN_NAME = 'timestamp'

DOWNLOAD_URL_MAP = {
    'ml-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-20m': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
}


def download_and_unzip_movielens_data(dataset: str, output_dir: Path, min_seq_length: int, min_user_feedback: int,
                                      min_item_feedback: int) -> (Path, Path):
    url = DOWNLOAD_URL_MAP[dataset]
    dataset_dir = output_dir / f'{dataset}_{min_seq_length}_{min_user_feedback}_{min_item_feedback}'
    download_dir = output_dir / dataset

    downloaded_file = download_dataset(url, download_dir)

    extract_dir = download_dir / 'raw'
    if not os.path.exists(extract_dir):
        extract_dir.mkdir(parents=True, exist_ok=True)

    unzip_file(downloaded_file, extract_dir, delete=False)
    return dataset_dir, extract_dir


def filter_ratings(ratings_df: pd.DataFrame,
                   min_user_feedback: int = 0,
                   min_item_feedback: int = 0
                   ):
    def _filter_dataframe(column: str, min_count: int, dataframe: pd.DataFrame) -> pd.DataFrame:
        sizes = ratings_df.groupby(column).size()
        good_entities = sizes.index[sizes >= min_count]

        return dataframe[ratings_df[column].isin(good_entities)]

    if min_user_feedback > 1:
        ratings_df = _filter_dataframe(RATING_USER_COLUMN_NAME, min_user_feedback, ratings_df)

    if min_item_feedback > 1:
        ratings_df = _filter_dataframe(RATING_MOVIE_COLUMN_NAME, min_item_feedback, ratings_df)

    return ratings_df


def preprocess_movielens_data(dataset_dir: Path,
                              output_dir: Path,
                              name: str,
                              delimiter: str = '\t',
                              min_user_feedback: int = 0,
                              min_item_feedback: int = 0
                              ) -> Path:
    """
    Convert raw movielens data to csv files and create vocabularies
    :param min_item_feedback:
    :param delimiter:
    :param output_dir:
    :param dataset_dir:
    :param name:
    :param min_user_feedback:
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

    movielens_1m = name == 'ml-1m'
    if movielens_1m:
        ratings_df.columns = [RATING_USER_COLUMN_NAME, RATING_MOVIE_COLUMN_NAME, 'rating', RATING_TIMESTAMP_COLUMN_NAME]

    ratings_df = filter_ratings(ratings_df,
                                min_user_feedback=min_user_feedback,
                                min_item_feedback=min_item_feedback)

    movies_df = read_csv(dataset_dir, "movies", file_type, sep, header)

    # only the ml-1m dataset has got a user info file …
    users_df = None

    if movielens_1m:
        movies_df.columns = ['movieId', 'title', 'genres']
        users_df = read_csv(dataset_dir, "users", file_type, sep, header)
        users_df.columns = [RATING_USER_COLUMN_NAME, 'gender', 'age', 'occupation', 'zip']
        ratings_df = pd.merge(ratings_df, users_df)
    elif name == "ml-20":
        links_df = read_csv(dataset_dir, "links", file_type, sep, header)
        ratings_df = pd.merge(ratings_df, links_df)

    merged_df = pd.merge(ratings_df, movies_df).sort_values(by=[RATING_USER_COLUMN_NAME, RATING_TIMESTAMP_COLUMN_NAME])
    # remove unused movie id column
    del merged_df[RATING_MOVIE_COLUMN_NAME]

    os.makedirs(output_dir, exist_ok=True)

    main_file = output_dir / f"{name}.csv"
    merged_df.to_csv(main_file, sep=delimiter, index=False)

    # build vocabularies
    build_vocabularies(merged_df, output_dir, "title")
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
                            min_seq_length: int = 3
                            ):
    # we use leave one out evaluation: the last watched movie for each users is in the test set, the second last is in
    # the valid set and the rest in the train set
    # we generate the session position index for validation and test (for train the validation index can be used
    # with the configuration that the target is not exposed to the training module)
    index_file = dataset_dir / f'{main_file.stem}.idx'

    index_csv(main_file, index_file, [session_key], delimiter)

    additional_features = {}

    create_conditional_index_using_extractor(main_file, index_file, dataset_dir / 'ml-1m.session.idx', item_header,
                                             min_seq_length, delimiter, additional_features, _all_remaining_positions)

    create_conditional_index_using_extractor(main_file, index_file, dataset_dir / 'ml-1m.validation.loo.idx', item_header,
                                             min_seq_length, delimiter, additional_features,
                                             functools.partial(_get_position_with_offset, offset=2))
    create_conditional_index_using_extractor(main_file, index_file, dataset_dir / 'ml-1m.test.loo.idx', item_header,
                                             min_seq_length, delimiter, additional_features,
                                             functools.partial(_get_position_with_offset, offset=1))


def _all_remaining_positions(session: Dict[str, Any]
                             ) -> Iterable[int]:
    return range(1, len(session[ITEM_SEQ_ENTRY_NAME]) - 2)
