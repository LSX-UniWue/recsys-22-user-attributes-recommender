from pathlib import Path

from asme.init.context import Context
from data.datamodule.config import DatasetConfig
from data.datamodule.preprocessing import ConvertToCsv, TransformCsv, CreateSessionIndex, \
    GroupAndFilter, GroupedFilter, CreateVocabulary, DELIMITER_KEY, EXTRACTED_DIRECTORY_KEY, \
    OUTPUT_DIR_KEY, CreateRatioSplit, CreateNextItemIndex, CreateLeaveOneOutSplit, CreatePopularity
from data.datamodule.converters import YooChooseConverter, Movielens1MConverter
from data.datamodule.extractors import RemainingSessionPositionExtractor
from data.datamodule.unpacker import Unzipper
from data.datamodule.preprocessing import PREFIXES_KEY
from datasets.vocabulary.create_vocabulary import ColumnInfo


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

    special_tokens = ["<PAD>", "<MASK>", "<UNK>"]
    columnnames = ["rating", "gender", "age", "occupation", "zip", "title", "genres"]
    delimiter = [None, None, None, None, None, None, "|"]
    columns = []
    for i, column in enumerate(columnnames):
        columns.append(ColumnInfo(column, delimiter[i]))

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
    return DatasetConfig(prefix,
                         "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
                         output_directory,
                         Unzipper(extraction_directory),
                         preprocessing_actions,
                         context)