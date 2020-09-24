from argparse import ArgumentParser
from pathlib import Path
from typing import Union, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets.nextitem import NextItemIndex, NextItemIterableDataset
from data.datasets.session import ItemSessionParser, ItemSessionDataset
from data.mp import mp_worker_init_fn
from data.utils import create_indexed_header, read_csv_header
from padding import padded_session_collate


class Dota2Small(pl.LightningDataModule):

    def __init__(self, local_path: str, remote_uri: str = None, delimiter="\t", max_seq_length: int = 2047,
                 batch_size: int = 64):

        super().__init__()

        self.local_path = Path(local_path)
        self.remote_uri = remote_uri
        self.delimiter = delimiter
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        train_data_file_path = self.local_path / "train.csv"
        train_index_file_path = self.local_path / "train.idx"
        train_nip_index_file_path = self.local_path / "train.nip.idx"

        train_dataset = self.create_dataset(train_data_file_path, train_index_file_path, train_nip_index_file_path, self.delimiter)
        training_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=padded_session_collate(self.max_seq_length),
            num_workers=1,
            worker_init_fn=mp_worker_init_fn
        )

        return training_loader

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        data_file_path = self.local_path / "valid.csv"
        index_file_path = self.local_path / "valid.idx"
        nip_index_file_path = self.local_path / "valid.nip.idx"

        dataset = self.create_dataset(data_file_path, index_file_path, nip_index_file_path, self.delimiter)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=padded_session_collate(self.max_seq_length),
            num_workers=1,
            worker_init_fn=mp_worker_init_fn
        )

        return loader

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        data_file_path = self.local_path / "test.csv"
        index_file_path = self.local_path / "test.idx"
        nip_index_file_path = self.local_path / "test.nip.idx"

        dataset = self.create_dataset(data_file_path, index_file_path, nip_index_file_path, self.delimiter)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=padded_session_collate(self.max_seq_length),
            num_workers=1,
            worker_init_fn=mp_worker_init_fn
        )

        return loader

    @staticmethod
    def create_dataset(data_file_path: Path, index_file_path: Path, nip_index_file_path: Path, delimiter: str):
        reader_index = CsvDatasetIndex(index_file_path)
        reader = CsvDatasetReader(data_file_path, reader_index)
        parser = ItemSessionParser(
            create_indexed_header(
                read_csv_header(data_file_path, delimiter=delimiter)
            ),
            "item_id", delimiter=delimiter
        )
        session_dataset = ItemSessionDataset(reader, parser)
        nip_index = NextItemIndex(nip_index_file_path)
        dataset = NextItemIterableDataset(session_dataset, nip_index)

        return dataset
