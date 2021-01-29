import os
import typer
from pathlib import Path
from dataset.dataset_pre_processing.movielens import download_and_unzip_movielens_data, preprocess_movielens_data
from dataset.dataset_pre_processing.yoochoose import pre_process_yoochoose_dataset, YOOCHOOSE_CLICKS_FILE_NAME, \
    YOOCHOOSE_SESSION_ID_KEY, YOOCHOOSE_ITEM_ID_KEY
from dataset.app.commands import index_command, split_commands, popularity_command, vocabulary_command

app = typer.Typer()


@app.command()
def movielens(dataset: str = typer.Argument(..., help="ml-1m or ml-20m", show_choices=True),
              output_dir: Path = typer.Option("./dataset/", help='directory to save data'),
              min_seq_length: int = typer.Option(5, help='the minimum feedback the user must have')
              ) -> None:
    # FixMe min_seq_length influences nothing except naming of dataset_dir
    dataset_dir, extract_dir = download_and_unzip_movielens_data(dataset, output_dir, min_seq_length)
    preprocess_movielens_data(extract_dir, dataset_dir, dataset)


@app.command()
def yoochoose(input_dir: Path = typer.Argument("./dataset/yoochoose-data",
                                               help='directory path to the raw yoochoose data set'),
              output_dir_path: Path = typer.Argument("./dataset/yoochoose-processed/",
                                                     help='Output directory for indices, splits, and vocabulary.'),
              min_seq_length: int = typer.Option(5, help='The minimum length of a session for the next item split')
              ) -> None:
    """
    Typer CLI function which wraps handling of pre-processing, splitting, storing and indexing of the yoochoose
    data set. As a prerequisite the yoochoose data set has to have been downloaded and stored at the input_dir.

    :param input_dir: Directory under which the yoochoose data set files are stored.
    :param output_dir_path: Directory under which the processed data is stored after successful execution.
    :param min_seq_length: Minimum length of a session in order to be included in the next item split
    :return:
    """
    delimiter = ","
    # Check if input dir contains the correct data path and that the yoochoose dataset is downloaded
    if not os.path.isfile(input_dir / (YOOCHOOSE_CLICKS_FILE_NAME + '.dat')):
        print(input_dir / (YOOCHOOSE_CLICKS_FILE_NAME + '.dat'),
              "does not exist. Please download the yoochoose data set and move it into", input_dir, ".")
        print("See --help for more information.")
    else:
        # Pre-process yoochoose data
        print("Perform pre-processing...")
        preprocessed_data_filepath = pre_process_yoochoose_dataset(input_dir, output_dir_path)
        print("Indexing processed data...")
        session_index_path = output_dir_path.joinpath(YOOCHOOSE_CLICKS_FILE_NAME + '.idx')
        index_command.index_csv(data_file_path=preprocessed_data_filepath,
                                index_file_path=session_index_path,
                                session_key=[YOOCHOOSE_SESSION_ID_KEY],
                                delimiter=delimiter)
        print("Build vocabulary...")
        vocabulary_output_file_path: Path = output_dir_path.joinpath("vocabulary").joinpath("tokens.txt")
        vocabulary_command.build(item_header_name=YOOCHOOSE_ITEM_ID_KEY,
                                 data_file_path=preprocessed_data_filepath,
                                 session_index_path=session_index_path,
                                 vocabulary_output_file_path=vocabulary_output_file_path,
                                 delimiter=delimiter)
        print("Build popularity...")
        popularity_command.build(data_file_path=preprocessed_data_filepath,
                                 session_index_path=session_index_path,
                                 vocabulary_file_path=vocabulary_output_file_path,
                                 output_file_path=output_dir_path.joinpath("popularity/popularity.txt"),
                                 item_header_name=YOOCHOOSE_ITEM_ID_KEY,
                                 min_session_length=min_seq_length,
                                 delimiter=delimiter)
        print("Create ratios split...")
        ratio_split_output_dir_path = output_dir_path.joinpath("ratios_split")
        split_commands.ratios(data_file_path=preprocessed_data_filepath,
                              session_index_path=session_index_path,
                              output_dir_path=ratio_split_output_dir_path,
                              train_ratio=0.9,
                              validation_ratio=0.05,
                              testing_ratio=0.05,
                              seed=123456)
        for split in ["train", "test", "valid"]:
            data_file = ratio_split_output_dir_path.joinpath(split + ".csv")
            index_file = ratio_split_output_dir_path.joinpath(split + ".idx")
            index_command.index_csv(data_file_path=data_file, index_file_path=index_file,
                                    session_key=[YOOCHOOSE_SESSION_ID_KEY],delimiter=delimiter)

        print("Create next item split...")
        # FixMe creates test and valid but not train.idx (Leave last two items out)
        split_commands.next_item(data_file_path=preprocessed_data_filepath,
                                 session_index_path=session_index_path,
                                 output_dir_path=output_dir_path.joinpath("next_item_split"),
                                 minimum_session_length=min_seq_length,
                                 delimiter=delimiter,
                                 item_header=YOOCHOOSE_ITEM_ID_KEY)
