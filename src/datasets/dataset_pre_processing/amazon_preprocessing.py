import pandas as pd
from pathlib import Path
from datasets.dataset_pre_processing.utils import download_dataset, unzip_gzip_file

AMAZON_DOWNLOAD_URL_MAP = {
    "games": "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Video_Games_v1_00.tsv.gz",
    "beauty": "https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz"
}

AMAZON_ZIPPED_FILE_NAMES = {
    "games": "amazon_reviews_us_Video_Games_v1_00.tsv.gz",
    "beauty": "amazon_reviews_us_Beauty_v1_00.tsv.gz"
}

AMAZON_FILE_NAMES = {
    "games": "amazon_reviews_us_Video_Games_v1_00.tsv",
    "beauty": "amazon_reviews_us_Beauty_v1_00.tsv"
}

AMAZON_DELIMITER = "\t"
AMAZON_ITEM_ID = "product_id"
AMAZON_SESSION_ID = "customer_id"


def download_and_unzip_amazon_dataset(category: str, output_dir: Path):
    raw_data_dir = output_dir
    download_dataset(AMAZON_DOWNLOAD_URL_MAP[category], raw_data_dir)
    unzip_gzip_file(src_file=raw_data_dir.joinpath(AMAZON_ZIPPED_FILE_NAMES[category]),
                    dest_file=raw_data_dir.joinpath(AMAZON_FILE_NAMES[category]),
                    delete_src_file=False)
    return raw_data_dir.joinpath(AMAZON_FILE_NAMES[category])  # Remove .gz since its unzipped


def preprocess_amazon_dataset_for_indexing(raw_data_tsv_file_path: Path):
    raw_df = pd.read_csv(filepath_or_buffer=raw_data_tsv_file_path, delimiter=AMAZON_DELIMITER, error_bad_lines=False)
    raw_df = raw_df.sort_values(AMAZON_SESSION_ID)
    raw_df.to_csv(raw_data_tsv_file_path, sep=AMAZON_DELIMITER)
    return raw_data_tsv_file_path
