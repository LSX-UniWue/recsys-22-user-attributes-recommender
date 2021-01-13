# User Guide #
Pre-Requisites:
* Built the virtual environment as described in the [Project Overview](./project_overview.md)

First of all you need to activate your virtual environment. This is accomplished using one of the
two following commands

````bash
# Either
poetry shell
# or
source recommender/venv/recommender/bin/activate
````

The components of this framework can be executed using the [Runner](./../runner).

## Pre-Processing Data Sets ##
For all data sets a CLI is provided via [Typer](https://typer.tiangolo.com/).
### MovieLens Data Set ###
To download and pre-process the MovieLens data set use the following commands:
````bash
python -m dataset.movielens ml-1m
python -m runner.dataset.create_reader_index ./dataset/ml-1m/ml-1m.csv ./dataset/ml-1m/index.csv --session_key userId
python -m runner.dataset.create_csv_dataset_splits ./dataset/ml-1m/ml-1m.csv ./dataset/ml-1m/index.csv \
        ./dataset/ml-1m/splits/ "train;0.9" "valid;0.05" "test;0.05"
python -m runner.dataset.create_next_item_index ./dataset/ml-1m/splits/test.csv ./dataset/ml-1m/index.csv \
        ./dataset/ml-1m/splits/test.nip.csv movieId
````
This downloads the MovieLens data set and prepares the data split for next item recommendation.

### Yoochoose Data Set ###
Pre-Requisites:
* Downloaded the [yoochoose data set](https://www.kaggle.com/chadgostopp/recsys-challenge-2015/download)

## Training implemented Models ##
[comment]: <ToDo>
## Executing Trained Models ##
[comment]: <ToDo>