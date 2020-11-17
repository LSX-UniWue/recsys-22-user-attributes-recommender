# A repository containing various recommender models
## Development
* Install [Poetry](https://python-poetry.org)
* Clone the repository
* Build the development virtual environment: `poetry install`
* Enter the virtual environment: `poetry shell`

 
## Dataset
* Download MovieLens data, e.g.
``` python -m dataset.movielens ml-1m```
* Prepare dataset index
``` python -m ./dataset/ml-1m/ml-1m.csv ./dataset/ml-1m/index.csv --session_key userId ```
* Prepare next item index
``` python -m runner.dataset.create_next_item_index ./dataset/ml-1m/ml-1m.csv ./dataset/ml-1m/index.csv ./dataset/ml-1m/next_item.csv movieId ```
* Test/train/validation split
``` python -m runner.dataset.create_csv_dataset_splits ./dataset/ml-1m/ml-1m.csv ./dataset/ml-1m/index.csv ./dataset/ml-1m/splits/ "train;0.9" "valid;0.05" "test;0.05"
```