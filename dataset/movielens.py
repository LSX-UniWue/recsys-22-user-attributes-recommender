from dataset.utils import *
import pandas as pd
import typer

app = typer.Typer()

def preprocess_data(dataset_dir, name):
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

    #Read and merge data
    ratings_df = read_csv(dataset_dir, "ratings", file_type, sep, header)
    movies_df = read_csv(dataset_dir, "movies", file_type, sep, header)

    if name == "ml-1m":
        ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']
        movies_df.columns = ['movieId', 'title', 'genres']
        users_df = read_csv(dataset_dir, "users", file_type, sep, header)
        users_df.columns = ['userId', 'gender', 'age','occupation','zip']
        ratings_df = pd.merge(ratings_df, users_df)

    elif name == "ml-20":
        links_df = read_csv(dataset_dir, "links", file_type, sep, header)
        ratings_df = pd.merge(ratings_df, links_df)

    merged_df = pd.merge(ratings_df, movies_df).sort_values(by=["userId", "timestamp"])
    merged_df.to_csv("/".join([dataset_dir, name + ".csv"]), sep="\t", index=False)

    #build vocabularies
    build_vocabularies(movies_df, dataset_dir, "title")
    build_vocabularies(movies_df, dataset_dir, "genres", split="|")
    build_vocabularies(users_df, dataset_dir, "gender")
    build_vocabularies(users_df, dataset_dir, "age")
    build_vocabularies(users_df, dataset_dir, "occupation")
    build_vocabularies(users_df, dataset_dir, "zip")


def build_vocabularies(dataframe, dataset_dir, column, split = ""):
    """
    Build and write a vocabulary file
    :param dataframe: base dataframe
    :param dataset_folder: folder for saving file
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
    special_tokens = pd.DataFrame(["<PAD>","<MASK>","<UNK>"])
    title_vocab = title_vocab.append(special_tokens).reset_index(drop=True)
    title_vocab["id"] = title_vocab.index
    os.makedirs("/".join([dataset_dir, "vocab"]), exist_ok= True)
    title_vocab.to_csv(("/".join([dataset_dir, "vocab", column + ".vocab"])), index=False, sep="\t")


def read_csv(dataset_dir, file, file_type, sep, header=None):
    file_path = "/".join([dataset_dir, file+file_type])
    return pd.read_csv(file_path, sep=sep, header=header, engine="python")

@app.command()
def main(dataset: str = typer.Argument(..., help="ml-1m or ml-20m"),
         dir: str = typer.Option("./dataset/",help='directory to save data')):

    if dataset == "ml-1m":
        url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    elif dataset == "ml-20m":
        url = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    else:
        print("Invalid dataset name")
        exit()

    dataset_dir = dir + "/" +dataset

    file = maybe_download(url, dataset_dir)
    unzip_file(file, dir, True)
    preprocess_data(dataset_dir, dataset)

@app.command()
def main(dataset: str = typer.Argument(..., help="ml-1m or ml-20m"),
         dir: str = typer.Option("./dataset/",help='directory to save data')):

    if dataset == "ml-1m":
        url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    elif dataset == "ml-20m":
        url = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
    else:
        print("Invalid dataset name")
        exit()

    dataset_dir = dir + "/" +dataset

    file = maybe_download(url, dataset_dir)
    unzip_file(file, dir, True)
    preprocess_data(dataset_dir, dataset)

if __name__ == "__main__":
    app()
