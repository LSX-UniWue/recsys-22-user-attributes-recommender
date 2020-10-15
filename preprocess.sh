#!/bin/bash
BASE="/home/dallmann/uni/research/dota/datasets/small/preprocessed"
DATA_FILENAME="small.csv"

#python -m runner.create_reader_index "${BASE}/${DATA_FILENAME}" "${BASE}/${DATA_FILENAME}.idx" --session_key id hero_id
#python -m runner.create_csv_dataset_splits "${BASE}/${DATA_FILENAME}" "${BASE}/${DATA_FILENAME}.idx" "${BASE}" "train;0.9" "valid;0.05" "test;0.05"

for FN in "valid" "test" "train"
do
  #python -m runner.create_reader_index "${BASE}/${FN}.csv" "${BASE}/${FN}.csv.idx" --session_key id hero_id
  python -m runner.create_next_item_index "${BASE}/${FN}.csv" "${BASE}/${FN}.csv.idx" "${BASE}/${FN}.csv.nip" item_id --min_session_length 2
done;

