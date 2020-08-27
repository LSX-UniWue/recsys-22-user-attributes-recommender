from pathlib import Path
from argparse import ArgumentParser

import random
import time

from datasets.base.parser import ItemIdSessionParser
from datasets.base.reader import Index, CsvSessionDatasetReader


def run(data_path: Path, index_path: Path, limit: int):
    index = Index(index_path)
    dataset = CsvSessionDatasetReader(data_path, "\t", index, ItemIdSessionParser("item_id"))

    samples = random.sample(range(index.num_sessions()), limit)

    start = time.perf_counter()

    for idx in samples:
        s = dataset.get_session(idx)

    stop = time.perf_counter()

    print(f"n={limit}, total time: {stop - start: 06.2f}, sample/s: {limit / (stop - start): 06.2f}")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("data", type=str, help="path to dataset")
    parser.add_argument("index", type=str, help="path to index")
    parser.add_argument("--limit", type=int, default=5000, help="number of calls to make. Default: 100000")

    args = parser.parse_args()

    data_path = Path(args.data)
    index_path = Path(args.index)
    limit = int(args.limit)

    run(data_path, index_path, limit)