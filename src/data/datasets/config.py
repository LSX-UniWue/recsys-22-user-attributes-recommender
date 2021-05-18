from pathlib import Path

from asme.init.context import Context
from data.datamodule.config import DatasetConfig
from data.datamodule.preprocessing import ConvertToCsv, TransformCsv, CreateSessionIndex, \
    GroupAndFilter, GroupedFilter, CreateVocabulary, DELIMITER_KEY, EXTRACTED_DIRECTORY_KEY, \
    OUTPUT_DIR_KEY, CreateRatioSplit, CreateNextItemIndex, CreateLeaveOneOutSplit
from data.datamodule.converters import YooChooseConverter, Movielens1MConverter
from data.datamodule.extractors import RemainingSessionPositionExtractor
from data.datamodule.unpacker import Unzipper
from data.datamodule.preprocessing import PREFIXES_KEY
from datasets.dataset_index_splits.conditional_split import _build_target_position_extractor


def get_movielens_1m_config(output_directory: Path,
                            extraction_directory: Path,
                            min_item_feedback: int = 0,
                            min_sequence_length: int = 2,
                            ) -> DatasetConfig:
    context = Context()
    context.set(PREFIXES_KEY, ["ml-1m"])
    context.set(DELIMITER_KEY, "\t")
    context.set(EXTRACTED_DIRECTORY_KEY, extraction_directory)
    context.set(OUTPUT_DIR_KEY, output_directory)

    preprocessing_actions = [ConvertToCsv(Movielens1MConverter()),
                             # TransformCsv(main_file, main_file, functools.partial(
                             #    Movielens1MConverter.filter_ratings,
                             #    min_item_feedback=min_item_feedback,
                             #    min_user_feedback=min_user_feedback)),
                             GroupAndFilter("movieId", GroupedFilter("count", lambda v: v >= min_item_feedback)),
                             GroupAndFilter("userId", GroupedFilter("count", lambda v: v >= min_sequence_length)),
                             CreateSessionIndex(["userId"]),
                             CreateVocabulary(["rating", "gender", "age", "occupation", "zip", "title", "genres"]),
                             CreateRatioSplit(0.8, 0.1, 0.1,
                                              [CreateSessionIndex(["userId"]),
                                               CreateNextItemIndex("title", RemainingSessionPositionExtractor(min_sequence_length))]),
                             CreateLeaveOneOutSplit("title")]
    return DatasetConfig("ml-1m",
                         "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
                         output_directory,
                         Unzipper(extraction_directory),
                         preprocessing_actions,
                         context)