# Dataset CLI

The pre-processing of data sets in this project is done with the help of a CLI which provides the following
functionality:
0. (Downloading of data sets not available for all data sets)
1. Pre-processing of data 
2. Indexing of data 
3. Splitting of the data set into train, test and validation split


The structure of the CLI (dataset_app.py) is as follows: 
- index
    - index-csv
- pre_process
    - movielens
    - yoochoose
    - amazon
- split   
    - next-item
    - ratios
- vocabulary
    - build
- popularity
    - build
    
## Usage

````bash
python main.py [OPTIONS] COMMAND [ARGS]...
````
Possible commands and sub-commands are listed above.

For example to generate the MovieLens 1m dataset, execute the following command:

````bash
python main.py pre_process movielens ml-1m
````

## Packaging
In the future it might be interesting to look at how this could be packaged using poetry 
(https://typer.tiangolo.com/tutorial/package/).