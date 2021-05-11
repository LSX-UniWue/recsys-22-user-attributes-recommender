import os
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd

from data.base.csv_index_builder import CsvSessionIndexer
from datasets.dataset_pre_processing.utils import read_csv


class YooChooseConverter:
    @staticmethod
    def to_csv(location: Path, output_file: Path):
        YOOCHOOSE_SESSION_ID_KEY = "SessionId"
        YOOCHOOSE_ITEM_ID_KEY = "ItemId"

        data = pd.read_csv(location.joinpath('clicks.dat'),
                           sep=',',
                           header=None,
                           usecols=[0, 1, 2],
                           dtype={0: np.int32, 1: str, 2: np.int64},
                           names=['SessionId', 'TimeStr', 'ItemId'])

        data['Time'] = data.TimeStr.apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
        session_lengths = data.groupby(YOOCHOOSE_SESSION_ID_KEY).size()
        data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]
        item_supports = data.groupby(YOOCHOOSE_ITEM_ID_KEY).size()
        data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]
        session_lengths = data.groupby(YOOCHOOSE_SESSION_ID_KEY).size()
        data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= 2].index)]

        if not os.path.exists(output_file):
            output_file.parent.mkdir(parents=True, exist_ok=True)
        data = data.sort_values(YOOCHOOSE_SESSION_ID_KEY)
        data.to_csv(path_or_buf=output_file)


class Movielens1MConverter:
    RATING_USER_COLUMN_NAME = 'userId'
    RATING_MOVIE_COLUMN_NAME = 'movieId'
    RATING_TIMESTAMP_COLUMN_NAME = 'timestamp'

    @staticmethod
    def to_csv(location: Path, output_file: Path, delimiter="\t"):

        file_type = ".dat"
        header = None
        sep = "::"
        name = "ml-1m"
        location = location / name
        ratings_df = read_csv(location, "ratings", file_type, sep, header)

        ratings_df.columns = [Movielens1MConverter.RATING_USER_COLUMN_NAME,
                              Movielens1MConverter.RATING_MOVIE_COLUMN_NAME, 'rating',
                              Movielens1MConverter.RATING_TIMESTAMP_COLUMN_NAME]

        movies_df = read_csv(location, "movies", file_type, sep, header)

        movies_df.columns = ['movieId', 'title', 'genres']
        users_df = read_csv(location, "users", file_type, sep, header)
        users_df.columns = [Movielens1MConverter.RATING_USER_COLUMN_NAME, 'gender', 'age', 'occupation', 'zip']
        ratings_df = pd.merge(ratings_df, users_df)

        merged_df = pd.merge(ratings_df, movies_df).sort_values(
            by=[Movielens1MConverter.RATING_USER_COLUMN_NAME, Movielens1MConverter.RATING_TIMESTAMP_COLUMN_NAME])

        os.makedirs(output_file.parent, exist_ok=True)

        merged_df.to_csv(output_file, sep=delimiter, index=False)

    @staticmethod
    def filter_ratings(ratings_df: pd.DataFrame,
                       min_user_feedback: int = 0,
                       min_item_feedback: int = 0
                       ):
        def _filter_dataframe(column: str, min_count: int, dataframe: pd.DataFrame) -> pd.DataFrame:
            sizes = ratings_df.groupby(column).size()
            good_entities = sizes.index[sizes >= min_count]

            return dataframe[ratings_df[column].isin(good_entities)]

        # (AD) we adopt the order used in bert4rec preprocessing
        if min_item_feedback > 1:
            ratings_df = _filter_dataframe(Movielens1MConverter.RATING_MOVIE_COLUMN_NAME, min_item_feedback, ratings_df)

        if min_user_feedback > 1:
            ratings_df = _filter_dataframe(Movielens1MConverter.RATING_USER_COLUMN_NAME, min_user_feedback, ratings_df)

        return ratings_df


class PreprocessingAction:

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, location: Path) -> None:
        pass

    def __call_(self, location: Path) -> None:
        self.apply(location)


class ConvertToCsv(PreprocessingAction):

    def __init__(self, output_file: Path, converter: Callable[[Path, Path], None]):
        self.converter = converter
        self.output_file = output_file

    def name(self) -> str:
        return "Convert to CSV"

    def apply(self, location: Path) -> None:
        self.converter(location, self.output_file)


class TransformCsv(PreprocessingAction):

    def __init__(self, input_file: Path, output_file: Path, filter: Callable[[pd.DataFrame], pd.DataFrame],
                 delimiter="\t"):
        self.input_file = input_file
        self.output_file = output_file
        self.delimiter = delimiter
        self.filter = filter

    def name(self) -> str:
        return "Filtering CSV"

    def apply(self, location: Path) -> None:
        df = pd.read_csv(location / self.input_file, delimiter=self.delimiter)
        filtered = self.filter(df)
        filtered.to_csv(location / self.output_file, sep=self.delimiter)


class CreateSessionIndex(PreprocessingAction):

    def __init__(self, input_file: Path, output_file: Path, session_key: List[str], delimiter="\t"):
        self.input_file = input_file
        self.output_file = output_file
        self.delimiter = delimiter
        self.session_key = session_key

    def name(self) -> str:
        return "Creating session index"

    def apply(self, location: Path) -> None:
        csv_index = CsvSessionIndexer(delimiter=self.delimiter)
        csv_index.create(location / self.input_file, location / self.output_file, self.session_key)
