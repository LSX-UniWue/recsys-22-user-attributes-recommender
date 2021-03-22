import os
import typer
from pathlib import Path

from datasets.data_structures.dataset_metadata import DatasetMetadata
from datasets.dataset_pre_processing.generic import generic_process_dataset

from datasets.dataset_pre_processing.movielens_preprocessing import download_and_unzip_movielens_data, \
    preprocess_movielens_data, MOVIELENS_DELIMITER, MOVIELENS_ITEM_HEADER_NAME, MOVIELENS_SESSION_KEY
from datasets.dataset_pre_processing.yoochoose_preprocessing import pre_process_yoochoose_dataset, \
    YOOCHOOSE_CLICKS_FILE_NAME, YOOCHOOSE_SESSION_ID_KEY, YOOCHOOSE_ITEM_ID_KEY, YOOCHOOSE_BUYS_FILE_NAME, \
    YOOCHOOSE_DELIMITER
from datasets.dataset_pre_processing.amazon_preprocessing import download_and_unzip_amazon_dataset, AMAZON_ITEM_ID, \
    AMAZON_SESSION_ID, AMAZON_DELIMITER, preprocess_amazon_dataset_for_indexing

app = typer.Typer()


@app.command()
def movielens(dataset: str = typer.Argument(..., help="ml-1m or ml-20m", show_choices=True),
              output_dir: Path = typer.Option("./dataset/", help='directory to save data'),
              min_seq_length: int = typer.Option(5, help='the minimum feedback the user must have'),
              min_user_feedback: int = typer.Option(0, help='the minimum number of feedback a user must have'),
              min_item_feedback: int = typer.Option(0, help='the minimum number of feedback an item must have received')
              ) -> None:
    dataset_dir, extract_dir = download_and_unzip_movielens_data(dataset, output_dir,
                                                                 min_seq_length=min_seq_length,
                                                                 min_item_feedback=min_item_feedback,
                                                                 min_user_feedback=min_user_feedback)
    main_file = preprocess_movielens_data(extract_dir, dataset_dir, dataset, min_item_feedback=min_item_feedback,
                                          min_user_feedback=min_user_feedback)
    stats_columns = ["title", "genres"]
    if dataset == "ml-1m":
        stats_columns += ["gender", "age", "occupation", "zip"]

    custom_tokens = ["<PAD>", "<MASK>", "<UNK>"]

    dataset_metadata: DatasetMetadata = DatasetMetadata(
        data_file_path=main_file,
        session_key=[MOVIELENS_SESSION_KEY],
        item_header_name=MOVIELENS_ITEM_HEADER_NAME,
        delimiter=MOVIELENS_DELIMITER,
        custom_tokens=custom_tokens,
        stats_columns=stats_columns
    )
    generic_process_dataset(dataset_metadata=dataset_metadata, min_seq_length=min_seq_length)


@app.command()
def yoochoose(input_dir: Path = typer.Argument("./dataset/yoochoose-data",
                                               help='directory path to the raw yoochoose data set'),
              output_dir_path: Path = typer.Argument("./dataset/yoochoose/",
                                                     help='Output directory for indices, splits, and vocabulary.'),
              category: str = typer.Option(..., help="buys or clicks"),
              min_seq_length: int = typer.Option(5, help='The minimum length of a session for the next item split')
              ) -> None:
    """
    Handles pre-processing, splitting, storing and indexing of yoochoose data set.
    As a prerequisite the yoochoose data set has to have been downloaded and stored at the input_dir.

    :param input_dir: Directory under which the yoochoose data set files are stored.
    :param output_dir_path: Directory under which the processed data is stored after successful execution.
    :param category: buys or clicks depending on which data should be processed
    :param min_seq_length: Minimum length of a session in order to be included in the next item split
    :return:
    """
    if category == "buys":
        file_name = YOOCHOOSE_BUYS_FILE_NAME
    elif category == "clicks":
        file_name = YOOCHOOSE_CLICKS_FILE_NAME
    else:
        raise Exception("Please specify a correct --category=[clicks or buys]")
    output_dir_path = output_dir_path / category
    # Check if input dir contains the correct data path and that the yoochoose dataset is downloaded
    if not os.path.isfile(input_dir / (file_name + '.dat')):
        print(input_dir / (file_name + '.dat'),
              "does not exist. Please download the yoochoose data set and move it into", input_dir, ".")
        print("See --help for more information.")
    else:
        # Pre-process yoochoose data
        print("Perform pre-processing...")
        preprocessed_data_filepath = pre_process_yoochoose_dataset(input_dir, output_dir_path, file_name=file_name)

        print("Creating necessary files for training and evaluation...")
        dataset_metadata: DatasetMetadata = DatasetMetadata(
            data_file_path=preprocessed_data_filepath,
            session_key=[YOOCHOOSE_SESSION_ID_KEY],
            item_header_name=YOOCHOOSE_ITEM_ID_KEY,
            delimiter=YOOCHOOSE_DELIMITER,
        )
        generic_process_dataset(dataset_metadata=dataset_metadata,
                                min_seq_length=min_seq_length)


@app.command()
def amazon(output_dir_path: Path = typer.Argument("./dataset/amazon/",
                                                  help='Output directory for indices, splits, and vocabulary.'),
           category: str = typer.Option(..., help="beatuy or games"),
           min_seq_length: int = typer.Option(5, help='The minimum length of a session for the next item split')
           ) -> None:
    """
    Handles pre-processing, splitting, storing and indexing of amazon data sets.

    :param output_dir_path: Directory under which the processed data is stored after successful execution.
    :param category: Amazon reviews category
    :param min_seq_length: Minimum length of a session in order to be included in the next item split
    :return:
    """
    output_dir_path = output_dir_path / category
    # Pre-process yoochoose data
    print("Download dataset...")
    raw_data_file_path = download_and_unzip_amazon_dataset(category=category, output_dir=output_dir_path)
    print("Pre-process data...")
    processed_data_file_path: Path = preprocess_amazon_dataset_for_indexing(
        raw_data_tsv_file_path=raw_data_file_path)
    print("Creating necessary files for training and evaluation...")
    dataset_metadata: DatasetMetadata = DatasetMetadata(
        data_file_path=processed_data_file_path,
        session_key=[AMAZON_SESSION_ID],
        item_header_name=AMAZON_ITEM_ID,
        delimiter=AMAZON_DELIMITER,
    )
    generic_process_dataset(dataset_metadata=dataset_metadata, min_seq_length=min_seq_length)
