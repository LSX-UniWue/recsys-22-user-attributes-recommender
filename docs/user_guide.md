# User Guide
Pre-Requisites:
* Built the virtual environment as described in the [README.md](../README.md)

First of all you need to activate your virtual environment. This is accomplished using one of the
two following commands:

````bash
# Either
poetry shell
# or
source recommender/venv/recommender/bin/activate
````

## Pre-Processing Datasets

###### Prerequisite: Make sure you are in the [src](./../src) folder
For all datasets a CLI is provided via [Typer](https://typer.tiangolo.com/).

Additional arguments (e.g. minimum sequence length or split ratio) are listed in the [Dataset CLI](./../docs/dataset/Dataset%20CLI.md).

### MovieLens Dataset ###
To download and pre-process the MovieLens dataset use the following command:
````bash
python datasets/main.py pre_process movielens [dataset] [ARGS]
````
with \[dataset] being either ````ml-1m```` or ````ml-20m````.

### Yoochoose Dataset ###
Pre-Requisites:
* Downloaded the [Yoochoose dataset](https://www.kaggle.com/chadgostopp/recsys-challenge-2015/download)
* Place the dataset in the ````./dataset/yoochoose-data```` folder

For pre-processing the Yoochoose dataset, use the following command:
````bash
python datasets/main.py pre_process yoochoose [category] [ARGS]
````
with \[category] being either ````buys```` or ````clicks````.

### Amazon Dataset ###
To download and pre-process the Amazon dataset use the following command:
````bash
python datasets/main.py pre_process amazon [category] [ARGS]
````
with \[category] being either ````beauty```` or ````games````.

## Training Implemented Models ##
````bash
python asme/main.py train [config_file]
````
with \[config_file] being the path to the jsonnet config file.

Additionally, ````do_test```` can be flagged to test the model after training.

### Resume Training ###
````bash
python asme/main.py resume [log_directory]
````
optional: \[checkpoint_file]

## Executing Trained Models ##

### Predicting ###
````bash
python asme/main.py predict [config_file] [checkpoint_file] [output_file]
````

### Evaluate ###
````bash
python asme/main.py evaluate [config_file] [checkpoint_file]
````

## Searching ##
````bash
python asme/main.py search [template_file] [study_name] [study_storage] [objective_metric] [study_directions] [num_trails]
````
with
* \[template_file]: the path to the config file
* \[study_name]: the name of an existing optuna study
* \[study_storage]: the connection string for the study storage
* \[objective_metric]: the name of the metric (e.g. recall@5)
* \[study_directions]: ````minimize```` or ````maximize````
* \[num_trails]: the number of trails to execute