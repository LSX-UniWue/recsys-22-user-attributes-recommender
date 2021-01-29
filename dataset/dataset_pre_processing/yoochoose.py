import typer
import os
from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt
from typing import Dict, List
from tqdm import tqdm

from data.base.indexer import CsvSessionIndexer
from dataset.dataset_pre_processing.utils import build_vocabularies

app = typer.Typer()
FULL_TRAIN_SET = "full_training_set"
TRAIN_SET = "training_set"
VALIDATION_SET = "validation_set"
TEST_SET = "test_set"

YOOCHOOSE_SESSION_ID_KEY = "SessionId"
YOOCHOOSE_ITEM_ID_KEY = "ItemId"
YOOCHOOSE_CLICKS_FILE_NAME = "yoochoose-clicks"


def pre_process_yoochoose_dataset(path_to_original_data: Path, output_dir_path: Path) -> Path:
    """
    Perform pre-processing for yoochoose data set as specified by Hidasi et al. 2016. Code adapted from
    https://github.com/hidasib/GRU4Rec/blob/master/examples/rsc15/preprocess.py.

    :param path_to_original_data: path to clicks.dat file
    :param output_dir_path: output dir for pre-processed csv
    :return: Stores preprocessed data at output_dir
    """
    data = pd.read_csv(path_to_original_data.joinpath(YOOCHOOSE_CLICKS_FILE_NAME+'.dat'),
                       sep=',',
                       header=None,
                       usecols=[0, 1, 2],
                       dtype={0: np.int32, 1: str, 2: np.int64},
                       names=['SessionId', 'TimeStr', 'ItemId'])

    data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
    session_lengths = data.groupby(YOOCHOOSE_SESSION_ID_KEY).size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]
    item_supports = data.groupby(YOOCHOOSE_ITEM_ID_KEY).size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]
    session_lengths = data.groupby(YOOCHOOSE_SESSION_ID_KEY).size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= 2].index)]

    if not os.path.exists(output_dir_path):
        output_dir_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir_path.joinpath(YOOCHOOSE_CLICKS_FILE_NAME+'.csv')
    data.to_csv(path_or_buf=output_file_path)
    return output_file_path


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DEPRECATED
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


def chronological_split(data: pd.DataFrame, output_dir: str) -> (Dict[str, pd.DataFrame], Path):
    """
    Split yoochoose data set according to Hidasi et al. 2016. Code adapted from
    https://github.com/hidasib/GRU4Rec/blob/master/examples/rsc15/preprocess.py.
    This split uses the last two days for validation and testing and the rest for training.

    :param data: preprocessed data frame containing yoochoose clicks data
    :param output_dir: path that the data is later stored at
    :return: Full train, train, validation and test split as dataframes in a dictionary.
    """
    # Create train_full set and testing set
    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_test = session_max_times[session_max_times >= tmax - 86400].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]
    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength >= 2].index)]

    # Create training and validation set
    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax - 86400].index
    session_valid = session_max_times[session_max_times >= tmax - 86400].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_valid)]
    valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]
    tslength = valid.groupby('SessionId').size()
    valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]

    result_dict = {FULL_TRAIN_SET: train,
                   TRAIN_SET: train_tr,
                   VALIDATION_SET: valid,
                   TEST_SET: test}
    print("result_dict", result_dict)
    return result_dict, Path(output_dir) / "chronological_split"


def sequential_split(data: pd.DataFrame, session_id_key: str, output_dir: str) -> (Dict[str, pd.DataFrame], Path):
    """
    Split yoochoose data sequentially, i.e., From every click sequence with length k use sequence[0:k-2] for training,
    sequence[-2] for validation and sequence[-1] for testing.

    :param data: pre-processed yoochoose data set
    :param session_id_key: data_frame key which specifies the column under which the session identifier can be found
    :param output_dir: path that the data is later stored at
    :return: Dictionary of dataframes containing training, validation and testing data.
    """
    # Filter out all sessions that are shorter than three since the sequence split cannot be performed for them
    session_lengths = data.groupby(session_id_key).size()
    data = data[np.in1d(data[session_id_key], session_lengths[session_lengths > 2].index)]
    # Group remaining sessions by ID and
    sessions = data.groupby(session_id_key).groups
    training_indices: List[int] = []
    validation_indices: List[int] = []
    testing_indices: List[int] = []
    # FIXME only use sessions where last two items exist in training part of any session
    for session_number, session_indices in sessions.items():
        session_indices = session_indices.tolist()
        training_indices += session_indices[:-2]

        val_index = session_indices[-2]
        test_index = session_indices[-1]
        validation_indices.append(val_index)
        testing_indices.append(test_index)
    return {TRAIN_SET: data.loc[training_indices],
            VALIDATION_SET: data.loc[validation_indices],
            TEST_SET: data.loc[testing_indices]}, Path(output_dir) / "sequential_split"


def write_splits_as_csv_and_create_index(split_data: Dict[str, pd.DataFrame], processed_data_dir_path: Path):
    for key, data_subset in tqdm(split_data.items(), desc="Create CSV-files and corresponding index for split."):
        csv_path: Path = processed_data_dir_path / (key + ".csv")
        index_path: Path = processed_data_dir_path / (key + ".idx")
        data_subset.to_csv(path_or_buf=csv_path)
        index = CsvSessionIndexer(delimiter=",")
        index.create(csv_path, index_path, [YOOCHOOSE_SESSION_ID_KEY])


def _yoochoose_old(input_dir: str = typer.Option("./yoochoose-data", help='directory path to the raw yoochoose data set'),
         output_dir: str = typer.Option("./yoochoose-processed/", help='directory to save data'),
         split: str = typer.Option("chronological", help='sequential or chronological')) -> None:
    """
    Typer CLI function which wraps handling of pre-processing, splitting, storing and indexing of the yoochoose
    data set. As a prerequisite the yoochoose data set has to have been downloaded and stored at the input_dir.

    :param input_dir: Directory under which the yoochoose data set files are stored.
    :param output_dir: Directory under which the processed data is stored after successful execution.
    :param split:
    :return:
    """
    # Check if input dir contains the correct data path and that the yoochoose dataset is downloaded
    input_dir = Path(input_dir)
    if not os.path.isdir(input_dir):
        print("Directory", input_dir,
              "does not exist.\nPlease specify the correct directory under which",
              YOOCHOOSE_CLICKS_FILE_NAME, "can be found.")
        print("See --help for more information.")
    elif not os.path.isfile(input_dir / YOOCHOOSE_CLICKS_FILE_NAME):
        print(input_dir / YOOCHOOSE_CLICKS_FILE_NAME,
              "does not exist. Please download the yoochoose data set and move it into",
              input_dir, ".")
        print("See --help for more information.")
    else:
        # Pre-process yoochoose data
        print("Perform pre-processing...")
        preprocessed_data = pre_process_yoochoose_dataset(Path(input_dir))
        # ToDo test the building of vocabulary
        build_vocabularies(preprocessed_data, input_dir, YOOCHOOSE_SESSION_ID_KEY)
        # Check if a valid split is specified
        print("Split data (%s)into train, validation and testing sets...", split)
        if split == "chronological":
            split_data, output_dir = chronological_split(data=preprocessed_data, output_dir=output_dir)
        elif split == "sequential":
            split_data, output_dir = sequential_split(data=preprocessed_data, session_id_key=YOOCHOOSE_SESSION_ID_KEY,
                                                      output_dir=output_dir)
        else:
            print("Split option has to be \"chronological\" or \"sequential\". You however specified: \"", split, "\".")
            print("See --help for more information.")
            split_data = None

        # If a valid split was performed create an index from it
        if split_data is not None:
            # Create output directory if it does not already exist.
            if not os.path.exists(output_dir):
                output_dir.mkdir(parents=True, exist_ok=True)
            # write the split data as .csv files and create respective indices
            write_splits_as_csv_and_create_index(split_data=split_data, processed_data_dir_path=output_dir)
