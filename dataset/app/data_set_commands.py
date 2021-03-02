import os
import typer
from pathlib import Path
from dataset.dataset_pre_processing.movielens import download_and_unzip_movielens_data, preprocess_movielens_data, \
    split_movielens_dataset
from dataset.dataset_pre_processing.yoochoose import pre_process_yoochoose_dataset, YOOCHOOSE_CLICKS_FILE_NAME, \
    YOOCHOOSE_SESSION_ID_KEY, YOOCHOOSE_ITEM_ID_KEY, YOOCHOOSE_BUYS_FILE_NAME
from dataset.dataset_pre_processing.amazon import download_and_unzip_amazon_dataset, AMAZON_ITEM_ID, \
    AMAZON_SESSION_ID, AMAZON_DELIMITER, preprocess_amazon_dataset_for_indexing
from dataset.app import split_commands, popularity_command, vocabulary_command, index_command

app = typer.Typer()


@app.command()
def movielens(dataset: str = typer.Argument(..., help="ml-1m or ml-20m", show_choices=True),
              output_dir: Path = typer.Option("./dataset/", help='directory to save data'),
              min_seq_length: int = typer.Option(5, help='the minimum feedback the user must have')
              ) -> None:
    # FixMe min_seq_length influences nothing except naming of dataset_dir
    dataset_dir, extract_dir = download_and_unzip_movielens_data(dataset, output_dir, min_seq_length)
    main_file = preprocess_movielens_data(extract_dir, dataset_dir, dataset)
    split_movielens_dataset(dataset_dir, main_file, min_seq_length=min_seq_length)


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
    delimiter = ","
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
        print("Indexing processed data...")
        session_index_path = output_dir_path.joinpath(file_name + '.session.idx')
        index_command.index_csv(data_file_path=preprocessed_data_filepath,
                                index_file_path=session_index_path,
                                session_key=[YOOCHOOSE_SESSION_ID_KEY],
                                delimiter=delimiter)
        print("Build vocabulary...")
        vocabulary_output_file_path: Path = output_dir_path.joinpath(file_name + ".vocabulary.txt")
        vocabulary_command.build(item_header_name=YOOCHOOSE_ITEM_ID_KEY,
                                 data_file_path=preprocessed_data_filepath,
                                 session_index_path=session_index_path,
                                 vocabulary_output_file_path=vocabulary_output_file_path,
                                 delimiter=delimiter)
        print("Build popularity...")
        popularity_output_file_path: Path = output_dir_path.joinpath(file_name + ".popularity.txt")
        popularity_command.build(data_file_path=preprocessed_data_filepath,
                                 session_index_path=session_index_path,
                                 vocabulary_file_path=vocabulary_output_file_path,
                                 output_file_path=popularity_output_file_path,
                                 item_header_name=YOOCHOOSE_ITEM_ID_KEY,
                                 min_session_length=min_seq_length,
                                 delimiter=delimiter)
        print("Create ratios split...")
        split_commands.ratios(data_file_path=preprocessed_data_filepath,
                              session_index_path=session_index_path,
                              output_dir_path=output_dir_path,
                              session_key=[YOOCHOOSE_SESSION_ID_KEY],
                              train_ratio=0.9,
                              validation_ratio=0.05,
                              testing_ratio=0.05,
                              delimiter=delimiter,
                              item_header_name=YOOCHOOSE_ITEM_ID_KEY,
                              minimum_session_length=min_seq_length,
                              seed=123456)

        print("Create next item split...")
        split_commands.next_item(data_file_path=preprocessed_data_filepath,
                                 session_index_path=session_index_path,
                                 output_dir_path=output_dir_path,
                                 minimum_session_length=min_seq_length,
                                 delimiter=delimiter,
                                 item_header=YOOCHOOSE_ITEM_ID_KEY)


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
    file_name: str = raw_data_file_path.stem
    print("Pre-process data...")
    preprocess_amazon_dataset_for_indexing(raw_data_tsv_file_path=raw_data_file_path)
    print("Indexing processed data...")
    session_index_path = raw_data_file_path.parent.joinpath(file_name+ '.session.idx')
    index_command.index_csv(data_file_path=raw_data_file_path,
                            index_file_path=session_index_path,
                            session_key=[AMAZON_SESSION_ID],
                            delimiter=AMAZON_DELIMITER)
    print("Build vocabulary...")
    vocabulary_output_file_path: Path = output_dir_path.joinpath(file_name + ".vocabulary.txt")
    vocabulary_command.build(item_header_name=AMAZON_ITEM_ID,
                             data_file_path=raw_data_file_path,
                             session_index_path=session_index_path,
                             vocabulary_output_file_path=vocabulary_output_file_path,
                             delimiter=AMAZON_DELIMITER)
    print("Build popularity...")
    popularity_command.build(data_file_path=raw_data_file_path,
                             session_index_path=session_index_path,
                             vocabulary_file_path=vocabulary_output_file_path,
                             output_file_path=output_dir_path.joinpath(file_name + ".popularity.txt"),
                             item_header_name=AMAZON_ITEM_ID,
                             min_session_length=min_seq_length,
                             delimiter=AMAZON_DELIMITER)
    print("Create ratios split...")
    split_commands.ratios(data_file_path=raw_data_file_path,
                          session_index_path=session_index_path,
                          output_dir_path=output_dir_path,
                          session_key=[AMAZON_SESSION_ID],
                          train_ratio=0.9,
                          validation_ratio=0.05,
                          testing_ratio=0.05,
                          delimiter=AMAZON_DELIMITER,
                          minimum_session_length=min_seq_length,
                          item_header_name=AMAZON_ITEM_ID,
                          seed=123456)

    print("Create next item split...")
    split_commands.next_item(data_file_path=raw_data_file_path,
                             session_index_path=session_index_path,
                             output_dir_path=output_dir_path,
                             minimum_session_length=min_seq_length,
                             delimiter=AMAZON_DELIMITER,
                             item_header=AMAZON_ITEM_ID)
