from pathlib import Path

from dataset.utils import maybe_download, unzip_file
import pandas as pd
import typer

app = typer.Typer()


DOWNLOAD_URL_MAP = {
    'ml-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    'ml-20m': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'
}


def preprocess_data(dataset_dir: Path,
                    name: str
                    ):
    """
    Convert raw movielens data to csv files and create vocabularies
    :param dataset_dir:
    :param name:
    :return:
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
    merged_df.to_csv(dataset_dir / f"{name}.csv", sep="\t", index=False)

    # build vocabularies
    build_vocabularies(movies_df, dataset_dir, "title")
    build_vocabularies(movies_df, dataset_dir, "genres", split="|")
    build_vocabularies(users_df, dataset_dir, "gender")
    build_vocabularies(users_df, dataset_dir, "age")
    build_vocabularies(users_df, dataset_dir, "occupation")
    build_vocabularies(users_df, dataset_dir, "zip")


def build_vocabularies(dataframe: pd.DataFrame,
                       dataset_dir: Path,
                       column: str,
                       split: str = ""
                       ) -> None:
    """
    Build and write a vocabulary file
    :param dataframe: base dataframe
    :param dataset_dir: folder for saving file
    :param dataset_name: dataset name for saving file
    :param column: column to create vocabulary for
    :param split: token to split if column need splitting
    :return:
    """
    if split != "":
        dataframe = pd.concat([pd.Series(row[column].split(split))
                               for _, row in dataframe.iterrows()]).reset_index()
        dataframe.columns = ['index', column]

    title_vocab = pd.DataFrame(dataframe[column].unique())
    special_tokens = pd.DataFrame(["<PAD>", "<MASK>", "<UNK>"])
    title_vocab = title_vocab.append(special_tokens).reset_index(drop=True)
    title_vocab["id"] = title_vocab.index

    vocab_file = dataset_dir / f'vocab_{column}.txt'
    title_vocab.to_csv(vocab_file, index=False, sep="\t")


def read_csv(dataset_dir: Path,
             file: str,
             file_type: str,
             sep: str,
             header: bool = None
             ) -> pd.DataFrame:
    file_path = dataset_dir / f"{file}{file_type}"
    return pd.read_csv(file_path, sep=sep, header=header, engine="python")


@app.command()
def main(dataset: str = typer.Argument(..., help="ml-1m or ml-20m", show_choices=True),
         output_dir: Path = typer.Option("./dataset/", help='directory to save data')
         ) -> None:

    url = DOWNLOAD_URL_MAP[dataset]
    dataset_dir = output_dir / dataset

    file = maybe_download(url, dataset_dir)
    unzip_file(file, output_dir, delete=False)
    preprocess_data(dataset_dir, dataset)


if __name__ == "__main__":
    app()
