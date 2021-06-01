# Dataset CLI

The pre-processing of datasets in this project is done with the help of a CLI which provides the following
functionality:

0. Downloading of datasets (not available for all datasets)
1. Pre-processing of data 
2. Indexing of data 
3. Splitting of the dataset into train, test and validation split


The command-structure of the CLI (dataset_app.py) is as follows: 
- pre_process
    - movielens
    - yoochoose
    - amazon
    - spotify
- index
    - index-csv
- split   
    - next-item
    - ratios
- vocabulary
    - build
- popularity
    - build
    
    
## Usage
###### Prerequisite: make sure you are in the [src](../../src) folder
````bash
python datasets/main.py [OPTIONS] COMMAND [ARGS]...
````
Possible commands and sub-commands are listed below.

### Pre-Process
Used to download (in case of amazon and movielens datasets) and pre-process different datasets, by executing multiple
CLI commands with dataset-specific values. E.g.:
````bash
python datasets/main.py pre_process amazon --category games --min-seq-length 5 --train-ratio 0.8 --validation-ratio 0.1 --test-ratio 0.1 --seed 1234
````

#### Movielens 
````bash
python datasets/main.py pre_process movielens <dataset> [OPTIONAL ARGS]
````

| argument | description | default value |
| ------ | ------ | ----- |
| dataset | 'ml-1m' or 'ml-20m' | - |
| output_dir | the directory | ./dataset/ |
| min_seq_length| the minimum feedback the user must have| 5 |
| min_user_feedback | the minimum number of feedback a user must have | 0 |
| min_item_feedback | the minimum number of feedback an item must have received | 0 |
| train_ratio | the share of sessions used for training | 0.9 |
| validation_ratio | the share of sessions used for validation | 0.05 |
| test_ratio | the share of sessions used for testing | 0.05 |
| seed | the seed for random sampling of splits | 123456 |

#### Yoochoose 
###### As a prerequisite the yoochoose dataset must be downloaded and stored in the input_dir
````bash
python datasets/main.py pre_process yoochoose <category> [OPTIONAL ARGS]
````
| argument | description | default value |
| ------ | ------ | ----- |
| category | 'buys' or 'clicks'| |
| input_dir | directory path to the raw yoochoose dataset | ./dataset/yoochoose-data |
| output_dir_path | the directory | './dataset/yoochoose' |
| min_seq_length| the minimum feedback the user must have| 5 |
| train_ratio | the share of sessions used for training | 0.9 |
| validation_ratio | the share of sessions used for validation | 0.05 |
| test_ratio | the share of sessions used for testing | 0.05 |
| seed | the seed for random sampling of splits | 123456 |

#### Amazon 
````bash
python datasets/main.py pre_process amazon <category> [OPTIONAL ARGS]
````
| argument | description | default value |
| ------ | ------ | ----- |
| category | 'beauty' or 'games'| |
| output_dir_path | the directory | './dataset/amazon' |
| min_seq_length| the minimum feedback the user must have| 5 |
| train_ratio | the share of sessions used for training | 0.9 |
| validation_ratio | the share of sessions used for validation | 0.05 |
| test_ratio | the share of sessions used for testing | 0.05 |
| seed | the seed for random sampling of splits | 123456 |

#### Spotify 
###### As a prerequisite the spotify dataset must be downloaded and stored in the input_data_path
````bash
python datasets/main.py pre_process spotify <input_data_path> [OPTIONAL ARGS]
````
| argument | description | default value |
| ------ | ------ | ----- |
| input_data_path | directory path to the raw spotify dataset | ./dataset/yoochoose-data |
| output_dir_path | the directory | './dataset/spotify-processed' |
| min_seq_length| the minimum feedback the user must have| 5 |
| train_ratio | the share of sessions used for training | 0.9 |
| validation_ratio | the share of sessions used for validation | 0.05 |
| test_ratio | the share of sessions used for testing | 0.05 |
| seed | the seed for random sampling of splits | 123456 |
### Index
Used to create a session index for a CSV-file where one entry for each session is written. E.g.:
````bash
python datasets/main.py index index-csv <input-csv-file>.csv <output-file>.idx <session-key> --delimiter $'\t'
````

### Split
Used to split a dataset into train, test, and validation splits. E.g.:
````bash
python datasets/main.py split next-item <input-csv>.csv <input-session-index>.idx <output-dir>
````

### Vocabulary
Used to create vocabulary for a dataset. E.g.:
````bash
python datasets/main.py vocabulary build <input-csv>.csv <input-session-index>.idx <output-vocab-file>.txt <item-key> --delimiter $'\t'
````
### Popularity
Used to create popularity for a dataset. E.g.:
````bash
python datasets/main.py popularity build <input-csv>.csv <input-session-index>.idx <input-vocab-file>.txt <output-popularity-file>.txt<item-key> --delimiter $'\t'
````

## Developer Guide
The files for the dataset CLI are located in [datasets](../../src/datasets). 
The folders under that directory contain the following contain the following files.

| Name | Implemented functionalities | files |
|---|---|---|
|[app](../../src/datasets/app)| Contains files necessary for the integration of functions into the Typer CLI | [data_set_commands.py](../../src/datasets/app/data_set_commands.py), [index_command.py](../../src/datasets/app/index_command.py), [split_commands.py](../../src/datasets/app/split_commands.py), [vocabulary_command.py](../../src/datasets/app/vocabulary_command.py), [popularity_command.py](../../src/datasets/app/popularity_command.py) |
|[data_structures](../../src/datasets/data_structures) | Contains helper classes which are used to simplify function calls and ensure correctness | [split_names.py](../../src/datasets/data_structures/split_names.py), [dataset_metadata.py](../../src/datasets/data_structures/dataset_metadata.py), [train_validation_test_splits_indices.py](../../src/datasets/data_structures/train_validation_test_splits_indices.py), [date_time_parser.py](../../src/datasets/data_structures/date_time_parser.py) | 
|[dataset_splits](../../src/datasets/dataset_splits)| Contains dataset specific pre-processing implementations as well as a [generic.py](../../src/datasets/dataset_pre_processing/generic.py) pre-processing that is used after for every dataset to integrate it into the ASME-framework | [generic.py](../../src/datasets/dataset_pre_processing/generic.py), [utils.py](../../src/datasets/dataset_pre_processing/utils.py), [yoochoose_preprocessing.py](../../src/datasets/dataset_pre_processing/yoochoose_preprocessing.py), [movielens_preprocessing.py](../../src/datasets/dataset_pre_processing/movielens_preprocessing.py), [amazon_preprocessing.py](../../src/datasets/dataset_pre_processing/amazon_preprocessing.py), [spotify_preprocessing.py](../../src/datasets/dataset_pre_processing/spotify_preprocessing.py) |
|[dataset_splits](../../src/datasets/dataset_splits)| Contains different implementations for the splitting of data sets and an [interface](../../src/datasets/dataset_splits/split_strategy.py) for such | [conditional_split.py](../../src/datasets/dataset_splits/conditional_split.py), [ratio_split_strategy.py](../../src/datasets/dataset_splits/ratio_split_strategy.py), [strategy_split.py](../../src/datasets/dataset_splits/strategy_split.py), [day_split_strategy.py](../../src/datasets/dataset_splits/day_split_strategy.py), [split_strategy.py](../../src/datasets/dataset_splits/split_strategy.py) |
|[popularity](../../src/datasets/popularity)|Contains implementation of counting the relative frequency of items in the dataset | [build_popularity.py](../../src/datasets/popularity/build_popularity.py) |
|[vocabulary](../../src/datasets/vocabulary)|Contains implementation of tokenization of items in a data set | [create_vocabulary.py](../../src/datasets/vocabulary/create_vocabulary.py) |
|[main.py](../../src/datasets/main.py)| Ties in the commands specified in [app](../../src/datasets/app) and is the entrypoint for Typer | - |


### Packaging
In the future it might be interesting to look at how this could be packaged using poetry 
(https://typer.tiangolo.com/tutorial/package/).
