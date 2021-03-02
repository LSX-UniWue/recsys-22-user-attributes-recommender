## Dataset Directory ##
### movielens.py: ###
- preprocess_data
	- Moved to dataset_preprocessing/movielens.py
- build vocabularies
	- Moved to dataset_preprocessing/utils.py
- read_csv
	- Moved to dataset_preprocessing/utils.py
- \_get_position_with_offset
	- Moved to datset_splits/conditional_split.py
- split_dataset
	- moved into dataset_preprocessing/movielens.py
- main
	- Reworked as movielens command in dataset/app/commands/data_set_commands.py
	- And as download_and_unzip_movielens_data() in dataset_preprocessing/movielens.py

### quick_indices_and_splits.py ###
- Unchanged

### utils.py (Deleted) ###
- download_dataset
	- Moved to dataset_preprocessing/utils.py
- unzip_file
	- Moved to dataset_preprocessing/utils.py


## Runner Directory ##
### stats/build_populatity_stats.py (Deleted) ###
- create_conditional_index moved to dataset/popularity/build_popularity.py and renamed as build
### create_conditional_index.py (Deleted)###
Moved into split commands.py:
- create_conditional_index
and dataset/dataset_splits/conditional_index.py:
- filter_by_sequence_feature
- \_build_target_position_extractor
- create_conditional_index_using_extractor

### create_csv_dataset_splits.py (Deleted) ###
- Has been reworked as ratios command under dataset/app/split_commands.py
- Code moved to dataset/dataset_splits/ratio_split.py

### create_reader_index (Deleted) ###
- Code moved to dataset/app/index_command.py
- create_index_for_csv renamed to index_csv
