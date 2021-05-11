import functools
from pathlib import Path

from data.datamodule.config import DatasetConfig
from data.datamodule.preprocessing import ConvertToCsv, TransformCsv, Movielens1MConverter, CreateSessionIndex
from data.datamodule.unpacker import Unzipper


def get_movielens_1m_config(output_directory: Path,
                            extraction_directory: Path,
                            min_user_feedback: int = 0,
                            min_item_feedback: int = 0
                            ) -> DatasetConfig:
    main_file = output_directory / "ml-1m.csv"
    session_index = output_directory / "ml-1m.session.idx"
    preprocessing_actions = [ConvertToCsv(main_file, Movielens1MConverter.to_csv),
                             TransformCsv(main_file, main_file, functools.partial(
                                 Movielens1MConverter.filter_ratings,
                                 min_item_feedback=min_item_feedback,
                                 min_user_feedback=min_user_feedback)),
                             CreateSessionIndex(main_file, session_index, ["userId"])]
    return DatasetConfig("ml-1m",
                         "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
                         output_directory,
                         Unzipper(extraction_directory),
                         preprocessing_actions)
