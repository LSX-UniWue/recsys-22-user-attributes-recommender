import os
import typer
import pandas as pd
from pathlib import Path

from dataset.app.utils import download_dataset, unzip_file, build_vocabularies, read_csv
from dataset.app.pre_process_command import app, DOWNLOAD_URL_MAP


@app.command()
def movielens(dataset: str = typer.Argument(..., help="ml-1m or ml-20m", show_choices=True),
         output_dir: Path = typer.Option("./dataset/", help='directory to save data'),
         min_seq_length: int = typer.Option(5, help='the minimum feedback the user must have')
         ) -> None:
    url = DOWNLOAD_URL_MAP[dataset]
    dataset_dir = output_dir / f'{dataset}_{min_seq_length}'
    download_dir = output_dir / dataset

    downloaded_file = download_dataset(url, download_dir)

    extract_dir = download_dir / 'raw'
    if not os.path.exists(extract_dir):
        extract_dir.mkdir(parents=True, exist_ok=True)

    unzip_file(downloaded_file, extract_dir, delete=False)
    preprocess_data(extract_dir, dataset_dir, dataset)


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

    os.makedirs(output_dir, exist_ok=True)

    main_file = output_dir / f"{name}.csv"
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

