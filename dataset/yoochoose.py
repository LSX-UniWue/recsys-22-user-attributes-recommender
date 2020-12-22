import typer
from pathlib import Path
import numpy as np
import pandas as pd
import datetime as dt
from typing import Dict, List

from runner.dataset.create_reader_index import create_index_for_csv

FULL_TRAIN_SET = "full_training_set"
TRAIN_SET = "training_set"
VALIDATION_SET = "validation_set"
TEST_SET = "test_set"

SESSION_ID_KEY = "SessionId"

app = typer.Typer()


def pre_process_yoochoose_dataset(path_to_original_data: Path) -> pd.DataFrame:
    data = pd.read_csv(path_to_original_data.joinpath('yoochoose-clicks.dat'),
                       sep=',',
                       header=None,
                       usecols=[0, 1, 2],
                       dtype={0: np.int32, 1: str, 2: np.int64},
                       names=['SessionId', 'TimeStr', 'ItemId'])

    data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= 2].index)]
    return data


def chronological_split(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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

    return {FULL_TRAIN_SET: train,
            TRAIN_SET: train_tr,
            VALIDATION_SET: valid,
            TEST_SET: test}


def sequential_split(data: pd.DataFrame, session_id_key: str) -> Dict[str, pd.DataFrame]:
    # Filter out all sessions that are shorter than three since the sequence split cannot be performed for them
    session_lengths = data.groupby(session_id_key).size()
    data = data[np.in1d(data[session_id_key], session_lengths[session_lengths > 2].index)]
    # Group remaining sessions by ID and
    sessions = data.groupby(session_id_key).groups
    training_indices: List[int] = []
    validation_indices: List[int] = []
    testing_indices: List[int] = []
    for session_number, session_indices in sessions.items():
        session_indices = session_indices.tolist()
        training_indices += session_indices[:-2]
        validation_indices.append(session_indices[-2])
        testing_indices.append(session_indices[-1])

    return {TRAIN_SET: data.loc[training_indices],
            VALIDATION_SET: data.loc[validation_indices],
            TEST_SET: data.loc[testing_indices]}


def write_splits_as_csv_and_create_index(split_data: Dict[str, pd.DataFrame], processed_data_dir_path: Path):
    for key, data_subset in split_data.items():
        csv_path: Path = processed_data_dir_path / (key + ".csv")
        index_path: Path = processed_data_dir_path / (key + ".idx")
        data_subset.to_csv(path_or_buf=csv_path)
        create_index_for_csv(input_path=csv_path, output_path=index_path, delimiter=',')


# TODO: Write stuff that has to do with the vocabulary
def write_vocab():
    pass


@app.command()
def main(input_dir: str = typer.Option("./yoochoose-data", help='directory to the raw yoochoose data set'),
         output_dir: str = typer.Option("./yoochoose-split/", help='directory to save data'),
         split: str = typer.Option("chronological", help='sequential or chronological')) -> None:
    preprocessed_data = pre_process_yoochoose_dataset(Path(input_dir))

    if split == "chronological":
        split_data = chronological_split(data=preprocessed_data)
    else:
        split_data = sequential_split(data=preprocessed_data, session_id_key=SESSION_ID_KEY)

    write_splits_as_csv_and_create_index(split_data=split_data, processed_data_dir_path=Path(output_dir))
