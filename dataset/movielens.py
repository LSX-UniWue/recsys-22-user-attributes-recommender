import functools
from pathlib import Path
from typing import Dict, Any, Iterable

from data.datasets import ITEM_SEQ_ENTRY_NAME
from dataset.utils import download_dataset, unzip_file
import pandas as pd
import typer

from runner.dataset.create_conditional_index import create_conditional_index_using_extractor
from runner.dataset.create_reader_index import create_index_for_csv

app = typer.Typer()


DOWNLOAD_URL_MAP = {
    'ml-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-20m': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
}


def preprocess_data(dataset_dir: Path,
                    output_dir: Path,
                    name: str,
                    delimiter: str = '\t'
                    ) -> Path:
    """
    Convert raw movielens data to csv files and create vocabularies
    :param delimiter:
    :param output_dir:
    :param dataset_dir:
    :param name:
    :return: the path to the main file
    """
    print("Convert to csv...")

    if name == "ml-1m":
        file_type = ".dat"
        header = None
        sep = "::"
    else:
        file_type = ".csv"
        header = 0
        sep = ","

    # read and merge data
    ratings_df = read_csv(dataset_dir, "ratings", file_type, sep, header)
    movies_df = read_csv(dataset_dir, "movies", file_type, sep, header)

    # only the ml-1m dataset has got a user info file …
    users_df = None

    if name == "ml-1m":
        ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']
        movies_df.columns = ['movieId', 'title', 'genres']
        users_df = read_csv(dataset_dir, "users", file_type, sep, header)
        users_df.columns = ['userId', 'gender', 'age', 'occupation', 'zip']
        ratings_df = pd.merge(ratings_df, users_df)

    elif name == "ml-20":
        links_df = read_csv(dataset_dir, "links", file_type, sep, header)
        ratings_df = pd.merge(ratings_df, links_df)

    merged_df = pd.merge(ratings_df, movies_df).sort_values(by=["userId", "timestamp"])
    # remove unused movie id column
    del merged_df['movieId']
    main_file = dataset_dir / f"{name}.csv"
    merged_df.to_csv(main_file, sep=delimiter, index=False)

    # build vocabularies
    # FIXME: the vocab should be build from the train set and not on the complete dataset
    build_vocabularies(movies_df, output_dir, "title")
    build_vocabularies(movies_df, output_dir, "genres", split="|")
    # for the ml-1m also export the vocabularies for the attributes
    if users_df is not None:
        build_vocabularies(users_df, output_dir, "gender")
        build_vocabularies(users_df, output_dir, "age")
        build_vocabularies(users_df, output_dir, "occupation")
        build_vocabularies(users_df, output_dir, "zip")

    return main_file


def build_vocabularies(dataframe: pd.DataFrame,
                       dataset_dir: Path,
                       column: str,
                       split: str = ""
                       ) -> None:
    """
    Build and write a vocabulary file
    :param dataframe: base dataframe
    :param dataset_dir: folder for saving file
    :param column: column to create vocabulary for
    :param split: token to split if column need splitting
    :return:
    """
    if split != "":
        dataframe = pd.concat([pd.Series(row[column].split(split))
                               for _, row in dataframe.iterrows()]).reset_index()
        dataframe.columns = ['index', column]

    title_vocab = pd.DataFrame(dataframe[column].unique())
    # we start with the pad token (pad token should have the id 0, so we start with the special tokens, than add
    # the remaining data)
    special_tokens = pd.DataFrame(["<PAD>", "<MASK>", "<UNK>"])
    title_vocab = special_tokens.append(title_vocab).reset_index(drop=True)
    title_vocab["id"] = title_vocab.index

    vocab_file = dataset_dir / f'vocab_{column}.txt'
    title_vocab.to_csv(vocab_file, index=False, sep="\t", header=False)


def read_csv(dataset_dir: Path,
             file: str,
             file_type: str,
             sep: str,
             header: bool = None
             ) -> pd.DataFrame:
    file_path = dataset_dir / f"{file}{file_type}"
    return pd.read_csv(file_path, sep=sep, header=header, engine="python")


def _get_position_with_offset(session: Dict[str, Any],
                              offset: int
                              ) -> Iterable[int]:
    sequence = session[ITEM_SEQ_ENTRY_NAME]
    return [len(sequence) - offset]


def split_dataset(dataset_dir: Path,
                  main_file: Path,
                  session_key: str = 'userId',
                  delimiter: str = '\t',
                  item_header: str = 'title',
                  min_seq_length: int = 4
                  ):
    # we use leave one out evaluation: the last watched movie for each users is in the test set, the second last is in
    # the valid test and the rest in the train set
    # we generate the session position index for validation and test (for train the validation index can be used the
    # validation index with the configuration that the target is not exposed to the model)
    index_file = dataset_dir / f'{main_file.stem}.idx'

    create_index_for_csv(main_file, index_file, [session_key], delimiter)

    additional_features = {}

    create_conditional_index_using_extractor(main_file, index_file, dataset_dir / 'valid.loo.idx', item_header,
                                             min_seq_length, delimiter, additional_features,
                                             functools.partial(_get_position_with_offset, offset=2))
    create_conditional_index_using_extractor(main_file, index_file, dataset_dir / 'test.loo.idx', item_header,
                                             min_seq_length, delimiter, additional_features,
                                             functools.partial(_get_position_with_offset, offset=1))


@app.command()
def main(dataset: str = typer.Argument(..., help="ml-1m or ml-20m", show_choices=True),
         output_dir: Path = typer.Option("./dataset/", help='directory to save data'),
         min_seq_length: int = typer.Option(5, help='the minimum feedback the user must have')
         ) -> None:
    url = DOWNLOAD_URL_MAP[dataset]
    dataset_dir = output_dir / f'f{dataset}_{min_seq_length}'
    download_dir = output_dir / dataset

    downloaded_file = download_dataset(url, download_dir)

    extract_dir = download_dir / 'raw'

    unzip_file(downloaded_file, extract_dir, delete=False)
    main_file = preprocess_data(extract_dir, dataset_dir, dataset)
    split_dataset(dataset_dir, main_file, min_seq_length=min_seq_length)


if __name__ == "__main__":
    app()
