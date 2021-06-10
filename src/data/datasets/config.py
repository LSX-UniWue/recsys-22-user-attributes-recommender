import copy
import inspect
from pathlib import Path
from typing import Callable

from asme.init.context import Context
from data.datamodule.config import DatasetPreprocessingConfig, register_preprocessing_config_provider, \
    PreprocessingConfigProvider
from data.datamodule.preprocessing import ConvertToCsv, TransformCsv, CreateSessionIndex, \
    GroupAndFilter, GroupedFilter, CreateVocabulary, DELIMITER_KEY, EXTRACTED_DIRECTORY_KEY, \
    OUTPUT_DIR_KEY, CreateRatioSplit, CreateNextItemIndex, CreateLeaveOneOutSplit, CreatePopularity
from data.datamodule.column_info import ColumnInfo
from data.datamodule.converters import YooChooseConverter, Movielens1MConverter
from data.datamodule.extractors import RemainingSessionPositionExtractor
from data.datamodule.unpacker import Unzipper
from data.datamodule.preprocessing import PREFIXES_KEY


def get_ml_1m_preprocessing_config(output_directory: str,
                                   extraction_directory: str,
                                   min_item_feedback: int = 0,
                                   min_sequence_length: int = 2,
                                   ) -> DatasetPreprocessingConfig:
    context = Context()
    context.set(PREFIXES_KEY, ["ml-1m"])
    context.set(DELIMITER_KEY, "\t")
    context.set(EXTRACTED_DIRECTORY_KEY, Path(extraction_directory))
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    special_tokens = ["<PAD>", "<MASK>", "<UNK>"]
    columns = [ColumnInfo("rating"), ColumnInfo("gender"), ColumnInfo("age"), ColumnInfo("occupation"),
               ColumnInfo("zip"), ColumnInfo("title"), ColumnInfo("genres"), ColumnInfo("genres", "|")]

    prefix = "ml-1m"
    preprocessing_actions = [ConvertToCsv(Movielens1MConverter()),
                             GroupAndFilter("movieId", GroupedFilter("count", lambda v: v >= min_item_feedback)),
                             GroupAndFilter("userId", GroupedFilter("count", lambda v: v >= min_sequence_length)),
                             CreateSessionIndex(["userId"]),
                             CreateRatioSplit(0.8, 0.1, 0.1,
                                              per_split_actions=
                                              [CreateSessionIndex(["userId"]),
                                               CreateNextItemIndex("title", RemainingSessionPositionExtractor(min_sequence_length))],
                                              complete_split_actions=
                                              [CreateVocabulary(columns, special_tokens=special_tokens, prefixes=[prefix]),
                                               CreatePopularity(columns, prefixes=[prefix])]),
                             CreateLeaveOneOutSplit("title",
                                                    inner_actions=
                                                    [CreateNextItemIndex("title", RemainingSessionPositionExtractor(min_sequence_length)),
                                                     CreateVocabulary(columns, special_tokens=special_tokens),
                                                     CreatePopularity(columns)])]
    return DatasetPreprocessingConfig(prefix,
                         "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
                                      Path(output_directory),
                                      Unzipper(Path(extraction_directory)),
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("ml-1m",
                                       PreprocessingConfigProvider(get_ml_1m_preprocessing_config,
                                                                   output_directory="./ml-1m",
                                                                   extraction_directory="./tmp/ml-1m",
                                                                   min_item_feedback=0,
                                                                   min_sequence_length=2))