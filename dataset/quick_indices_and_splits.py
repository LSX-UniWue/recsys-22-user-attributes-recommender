from dataset.utils import *
import pandas as pd
import typer
from pathlib import Path

from runner.dataset.create_reader_index import run as create_index
from runner.dataset.create_csv_dataset_splits import run as create_splits, extract_splits
from runner.dataset.create_conditional_index import run as create_conditional_index

app = typer.Typer()


@app.command()
def main(dataset: str = typer.Argument(..., help="ml-1m or ml-20m"),
         session_key: str = typer.Argument("userId", help="session key"),
         item_header: str = typer.Argument("title", help="item column"),
         dir: str = typer.Option("./dataset/",help='directory to save data'),
         seed: int = typer.Option(123456,help='seed for split'),
         train: float = typer.Option(0.9,help="train_split"),
         valid: float = typer.Option(0.05,help="train_split"),
         test: float = typer.Option(0.05,help="train_split"),
         min_session_length: int = typer.Option(2, help="minimum session length")
         ):

    dataset_dir = dir + "/" +dataset
    path_main_csv = Path("./"+dataset_dir+"/"+dataset+".csv")
    path_main_index = Path("./"+dataset_dir+"/"+dataset+".index")
    split_dir_path = Path("./"+dataset_dir+"/splits/")
    split_dir_path.mkdir(parents=True, exist_ok=True)

    create_index(path_main_csv, path_main_index, session_key=[session_key])

    splits = {"train": train, "valid": valid, "test": test}
    create_splits(path_main_csv, path_main_index, split_dir_path, splits, seed)

    for split in ["test","train","valid"]:
        split_path = Path("./"+dataset_dir+"/splits/"+split+".csv")
        split_path_index = Path("./"+dataset_dir+"/splits/"+split+".index")
        split_path_next_index = Path("./"+dataset_dir+"/splits/"+split+".nip")
        create_index(split_path, split_path_index, session_key=[session_key])
        create_conditional_index(data_file_path=split_path,
                                 session_index_path=split_path_index,
                                 output_file_path=split_path_next_index,
                                 item_header_name=item_header,
                                 min_session_length=min_session_length,
                                 delimiter="\t",target_feature=None)



if __name__ == "__main__":
    app()
