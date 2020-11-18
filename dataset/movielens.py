from dataset.utils import *
import pandas as pd
import typer

app = typer.Typer()




def convert_data_to_csv(dataset_folder,name):
    print("Convert to csv...")

    if name == "ml-1m":
        file_type = ".dat"
        header = None
        sep = "::"
    elif name == "ml-20m":
        file_type = ".csv"
        header = 0
        sep = ","

    ratings_df = read_csv(dataset_folder, name, "ratings", file_type, sep, header)
    movies_df = read_csv(dataset_folder, name, "movies", file_type, sep, header)

    if name == "ml-1m":
        ratings_df.columns = ['userId', 'movieId', 'rating', 'timestamp']
        movies_df.columns = ['movieId', 'title', 'genres']
        users_df = read_csv(dataset_folder, name, "users", file_type, sep, header)
        users_df.columns = ['userId', 'gender', 'age','occupation','zip']
        ratings_df = pd.merge(ratings_df,users_df)

    elif name == "ml-20":
        links_df = read_csv(dataset_folder, name, "links", file_type, sep, header)
        ratings_df = pd.merge(ratings_df, links_df)

    merged_df = pd.merge(ratings_df, movies_df).sort_values(by=["userId", "timestamp"])
    merged_df.to_csv("/".join([dataset_folder, name, name+".csv"]), sep="\t", index=False)
    movies_df["id"] = movies_df.index
    movies_df[['title','id']].to_csv("/".join([dataset_folder, name, name+".vocab"]),sep ="\t",index=False)


def read_csv(dataset_folder, name, file, file_type, sep, header=None):
    file_path = "/".join([dataset_folder, name, file+file_type])
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

    file = maybe_download(url, dir+"/"+dataset)
    unzip_file(file, dir, True)
    convert_data_to_csv(dir, dataset)


if __name__ == "__main__":
    app()








