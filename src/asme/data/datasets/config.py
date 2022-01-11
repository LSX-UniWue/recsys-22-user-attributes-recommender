from pathlib import Path

from asme.core.init.context import Context
from asme.data.datamodule.config import DatasetPreprocessingConfig, PreprocessingConfigProvider
from asme.data.datamodule.converters import YooChooseConverter, Movielens1MConverter, ExampleConverter, \
    Movielens20MConverter, AmazonConverter, SteamConverter
from asme.data.datamodule.extractors import RemainingSessionPositionExtractor, SlidingWindowPositionExtractor
from asme.data.datamodule.preprocessing.action import PREFIXES_KEY, DELIMITER_KEY, INPUT_DIR_KEY, OUTPUT_DIR_KEY
from asme.data.datamodule.preprocessing.csv import ConvertToCsv, CopyMainFile
from asme.data.datamodule.preprocessing.indexing import CreateSessionIndex, CreateNextItemIndex, \
    CreateSlidingWindowIndex
from asme.data.datamodule.preprocessing.popularity import CreatePopularity
from asme.data.datamodule.preprocessing.split import CreateRatioSplit, CreateLeaveOneOutSplit
from asme.data.datamodule.preprocessing.template import build_ratio_split, build_leave_one_out_split, \
    build_leave_percentage_out_split
from asme.data.datamodule.preprocessing.vocabulary import CreateVocabulary
from asme.data.datamodule.registry import register_preprocessing_config_provider
from asme.data.datamodule.unpacker import Unzipper
from asme.data.datasets.sequence import MetaInformation


def get_ml_1m_preprocessing_config(
        # General parameters
        output_directory: str,
        extraction_directory: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int,
) -> DatasetPreprocessingConfig:
    prefix = "ml-1m"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(extraction_directory))
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("rating", type="int", run_tokenization=False),
               MetaInformation("gender", type="str"),
               MetaInformation("age", type="int", run_tokenization=False),
               MetaInformation("occupation", type="str"),
               MetaInformation("zip", type="str"),
               MetaInformation("title", type="str"),
               MetaInformation("genres", type="str", configs={"delimiter": "|"})]

    item_column = MetaInformation("item", column_name="title", type="str")
    min_item_feedback_column = "movieId"
    min_sequence_length_column = "userId"
    session_key = ["userId"]

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(Movielens1MConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]

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
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4,
                                                                   window_markov_length=3,
                                                                   window_target_length=3,
                                                                   session_end_offset=0
                                                                   ))


def get_ml_20m_preprocessing_config(
        # General parameters
        output_directory: str,
        extraction_directory: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int,
) -> DatasetPreprocessingConfig:
    prefix = "ml-20m"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(extraction_directory))
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("rating", type="int", run_tokenization=False),
               MetaInformation("timestamp", type="str"),
               MetaInformation("title", type="str"),
               MetaInformation("genres", type="str", configs={"delimiter": "|"})]

    item_column = MetaInformation("item", column_name="title", type="str")
    min_item_feedback_column = "movieId"
    min_sequence_length_column = "userId"
    session_key = ["userId"]

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(Movielens20MConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]
    return DatasetPreprocessingConfig(prefix,
                                      "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
                                      Path(output_directory),
                                      Unzipper(Path(extraction_directory)),
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("ml-20m",
                                       PreprocessingConfigProvider(get_ml_20m_preprocessing_config,
                                                                   output_directory="./ml-20m",
                                                                   extraction_directory="./tmp/ml-20m",
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4))


def get_amazon_preprocessing_config(
        # General parameters
        prefix: str,
        output_directory: str,
        input_directory: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int,
) -> DatasetPreprocessingConfig:
    if prefix not in ["games", "beauty"]:
        raise KeyError("The only amazon datasets that are currently supported are 'games' and 'beauty'.")

    AMAZON_DOWNLOAD_URL_MAP = {
        "games": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games.json.gz",
        "beauty": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz"
    }

    AMAZON_ZIPPED_FILE_NAMES = {
        "games": "reviews_Video_Games.json.gz",
        "beauty": "reviews_Beauty.json.gz"
    }

    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(input_directory) / AMAZON_ZIPPED_FILE_NAMES[prefix])
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("reviewer_id", type="str"),
               MetaInformation("product_id", type="str"),
               MetaInformation("timestamp", type="int", run_tokenization=False)]

    min_item_feedback_column = "product_id"
    min_sequence_length_column = "reviewer_id"
    session_key = ["reviewer_id"]
    item_column = MetaInformation("item", column_name="product_id", type="str")

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(AmazonConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]

    return DatasetPreprocessingConfig(prefix,
                                      AMAZON_DOWNLOAD_URL_MAP[prefix],
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("beauty",
                                       PreprocessingConfigProvider(get_amazon_preprocessing_config,
                                                                   prefix="beauty",
                                                                   output_directory="./beauty",
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4))

register_preprocessing_config_provider("games",
                                       PreprocessingConfigProvider(get_amazon_preprocessing_config,
                                                                   prefix="games",
                                                                   output_directory="./games",
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4))


def get_steam_preprocessing_config(
        # General parameters
        output_directory: str,
        input_dir: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int
) -> DatasetPreprocessingConfig:
    prefix = "steam"
    filename = "steam_reviews.json.gz"

    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(input_dir) / filename)
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("username", type="str"),
               MetaInformation("product_id", type="str"),
               MetaInformation("date", type="timestamp", configs={"format": "%Y-%m-%d"}, run_tokenization=False)]

    min_item_feedback_column = "product_id"
    min_sequence_length_column = "usermname"
    session_key = ["username"]
    item_column = MetaInformation("item", column_name="product_id", type="str")

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(SteamConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]

    return DatasetPreprocessingConfig(prefix,
                                      "http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("steam",
                                       PreprocessingConfigProvider(get_steam_preprocessing_config,
                                                                   output_directory="./steam",
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4))


def get_yoochoose_preprocessing_config(
        # General parameters
        output_directory: str,
        input_directory: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int
) -> DatasetPreprocessingConfig:

    prefix = "yoochoose"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(INPUT_DIR_KEY, Path(input_directory))
    context.set(OUTPUT_DIR_KEY, Path(output_directory))

    columns = [MetaInformation("SessionId", type="str"),
               MetaInformation("ItemId", type="str"),
               MetaInformation("Time", type="int", run_tokenization=False)]

    min_item_feedback_column = "ItemId"
    min_sequence_length_column = "SessionId"
    session_key = ["SessionId"]
    item_column = MetaInformation("item", column_name="ItemId", type="str")

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(YooChooseConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]

    return DatasetPreprocessingConfig(prefix,
                                      None,
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("yoochoose",
                                       PreprocessingConfigProvider(get_yoochoose_preprocessing_config,
                                                                   output_directory="./yoochoose",
                                                                   min_item_feedback=4,
                                                                   min_sequence_length=4))

"""
hier müssen wir noch die Parameter ändern:
window_size setzt sich aus sequence length und target_length zusammen
min_input_length bei extractor??
außerdem bei LOO auch Sliding_window index??
"""


def get_example_preprocessing_config(
        # General parameters
        output_directory: str,
        input_file_path: str,
        # Ratio split parameters
        ratio_split_min_item_feedback: int,
        ratio_split_min_sequence_length: int,
        ratio_split_train_percentage: float,
        ratio_split_validation_percentage: float,
        ratio_split_test_percentage: float,
        ratio_split_window_markov_length: int,
        ratio_split_window_target_length: int,
        ratio_split_session_end_offset: int,
        # Leave one out split parameters
        loo_split_min_item_feedback: int,
        loo_split_min_sequence_length: int,
        # Leave percentage out split parameters
        lpo_split_min_item_feedback: int,
        lpo_split_min_sequence_length: int,
        lpo_split_train_percentage: float,
        lpo_split_validation_percentage: float,
        lpo_split_test_percentage: float,
        lpo_split_min_train_length: int,
        lpo_split_min_validation_length: int,
        lpo_split_min_test_length: int
) -> DatasetPreprocessingConfig:
    prefix = "example"
    context = Context()
    context.set(PREFIXES_KEY, [prefix])
    context.set(DELIMITER_KEY, "\t")
    context.set(OUTPUT_DIR_KEY, Path(output_directory))
    context.set(INPUT_DIR_KEY, input_file_path)

    # FIXME (AD) we're forced to set column_name because vocabulary and popularity code relies on it being set.
    columns = [MetaInformation("item_id", column_name="item_id", type="str"),
               # TODO (AD) find out why setting type to int prevents correct vocabulary creation (vocabulary is not saved with consecutive ids)
               MetaInformation("user_id", column_name="user_id", type="str"),
               MetaInformation("attr_one", column_name="attr_one", type="str")]

    min_item_feedback_column = "item_id"
    min_sequence_length_column = "session_id"
    session_key = ["session_id"]
    item_column = MetaInformation("item", column_name="item_id", type="str")

    ratio_split_action = build_ratio_split(columns, prefix, ratio_split_min_item_feedback, min_item_feedback_column,
                                           ratio_split_min_sequence_length, min_sequence_length_column,
                                           session_key, [item_column], ratio_split_train_percentage,
                                           ratio_split_validation_percentage, ratio_split_test_percentage,
                                           ratio_split_window_markov_length, ratio_split_window_target_length,
                                           ratio_split_session_end_offset)

    leave_one_out_split_action = build_leave_one_out_split(columns, prefix, loo_split_min_item_feedback,
                                                           min_item_feedback_column,
                                                           loo_split_min_sequence_length, min_sequence_length_column,
                                                           session_key, item_column, [item_column])

    leave_percentage_out_split_action = build_leave_percentage_out_split(columns, prefix, lpo_split_min_item_feedback,
                                                                         min_item_feedback_column,
                                                                         lpo_split_min_sequence_length,
                                                                         min_sequence_length_column,
                                                                         session_key, item_column, [item_column],
                                                                         lpo_split_train_percentage,
                                                                         lpo_split_validation_percentage,
                                                                         lpo_split_test_percentage,
                                                                         lpo_split_min_train_length,
                                                                         lpo_split_min_validation_length,
                                                                         lpo_split_min_test_length)

    preprocessing_actions = [ConvertToCsv(ExampleConverter()),
                             ratio_split_action, leave_one_out_split_action,
                             leave_percentage_out_split_action]

    return DatasetPreprocessingConfig(prefix,
                                      None,
                                      Path(output_directory),
                                      None,
                                      preprocessing_actions,
                                      context)


register_preprocessing_config_provider("example",
                                       PreprocessingConfigProvider(get_example_preprocessing_config,
                                                                   output_directory="./example",
                                                                   input_file_path="../tests/example_dataset/example.csv",
                                                                   min_sequence_length=3,
                                                                   window_markov_length=2,
                                                                   window_target_length=1,
                                                                   session_end_offset=0
                                                                   ))
