from pathlib import Path
from typing import Union, List, Optional
import shutil
import urllib.parse
import pytorch_lightning as pl
from pyhocon import ConfigTree
from torch.utils.data import DataLoader
from tqdm import tqdm
import tarfile

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from data.datasets.nextitem import NextItemIndex, NextItemIterableDataset, NextItemDataset
from data.datasets.session import ItemSessionParser, ItemSessionDataset
from data.mp import mp_worker_init_fn
from data.utils import create_indexed_header, read_csv_header
from padding import padded_session_collate
from tokenization.tokenizer import Tokenizer
from tokenization.vocabulary import VocabularyBuilder, CSVVocabularyReaderWriter


class Dota2Small(pl.LightningDataModule):
    CONFIG_PATH_KEY = "dataset.dota"
    TOKEN_VOCABULARY_FILENAME = "items.vocab"
    TRAIN_BASENAME = "train"
    VALID_BASENAME = "valid"
    TEST_BASENAME = "test"

    DATASET_SUFFIX = "csv"
    SESSION_INDEX_SUFFIX = "idx"
    NEXT_ITEM_INDEX_SUFFIX = "nip.idx"

    PAD_TOKEN_NAME = "<PAD>"

    def __init__(self,
                 local_path: str, 
                 remote_uri: str = None,
                 delimiter="\t",
                 max_seq_length: int = 2047,
                 batch_size: int = 64,
                 num_workers: int = 1):

        super().__init__()

        self.local_path = Path(local_path)
        self.remote_uri = remote_uri
        self.delimiter = delimiter
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers

    @staticmethod
    def from_configuration(config: ConfigTree):
        dota_config = config[Dota2Small.CONFIG_PATH_KEY]

        local_path = dota_config["local_path"]
        remote_uri = dota_config["remote_uri"]
        delimiter = dota_config["delimiter"]
        batch_size = dota_config.get_int("batch_size")
        max_seq_length = dota_config.get_int("max_seq_length")
        num_workers = dota_config.get_int("num_workers")

        return Dota2Small(local_path, remote_uri, delimiter, max_seq_length, batch_size, num_workers)

    def prepare_data(self, *args, **kwargs):
        self._copy_remote_archive()
        self._extract_archive()
        self._create_token_vocabulary()

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        tokenizer = self.load_tokenizer()
        train_dataset = self.create_next_item_dataset(Dota2Small.TRAIN_BASENAME, self.delimiter, tokenizer)

        training_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=padded_session_collate(self.max_seq_length, tokenizer.pad_token_id),
            num_workers=self.num_workers,
            worker_init_fn=mp_worker_init_fn
        )

        return training_loader

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        tokenizer = self.load_tokenizer()
        dataset = self.create_next_item_dataset(Dota2Small.VALID_BASENAME, self.delimiter, tokenizer, shuffle=False)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=padded_session_collate(self.max_seq_length, tokenizer.pad_token_id),
            num_workers=1,
            worker_init_fn=mp_worker_init_fn,
            shuffle=False,
            drop_last=False
        )

        return loader

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        tokenizer = self.load_tokenizer()
        dataset = self.create_next_item_dataset(Dota2Small.TEST_BASENAME, self.delimiter, tokenizer, shuffle=False)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=padded_session_collate(self.max_seq_length, tokenizer.pad_token_id),
            num_workers=1,
            worker_init_fn=mp_worker_init_fn,
            shuffle=False,
            drop_last=False
        )

        return loader

    def load_tokenizer(self) -> Tokenizer:
        item_vocabulary_path = self.local_path / Dota2Small.TOKEN_VOCABULARY_FILENAME
        with item_vocabulary_path.open("r") as file:
            vocabulary = CSVVocabularyReaderWriter().read(file)

        tokenizer = Tokenizer(vocabulary, pad_token=Dota2Small.PAD_TOKEN_NAME)

        return tokenizer

    def create_session_dataset(self, dataset_basename: str, delimiter: str, tokenizer: Tokenizer = None) -> ItemSessionDataset:
        data_file_path = self.local_path / f"{dataset_basename}.{Dota2Small.DATASET_SUFFIX}"
        index_file_path = self.local_path / f"{dataset_basename}.{Dota2Small.SESSION_INDEX_SUFFIX}"

        reader_index = CsvDatasetIndex(index_file_path)
        reader = CsvDatasetReader(data_file_path, reader_index)
        parser = ItemSessionParser(
            create_indexed_header(
                read_csv_header(data_file_path, delimiter=delimiter)
            ),
            "item_id", delimiter=delimiter
        )

        session_dataset = ItemSessionDataset(reader, parser, tokenizer=tokenizer)

        return session_dataset

    def create_next_item_dataset(self, dataset_basename: str, delimiter: str, tokenizer: Tokenizer, shuffle: bool = True):
        session_dataset = self.create_session_dataset(dataset_basename, delimiter, tokenizer)

        nip_index_file_path = self.local_path / f"{dataset_basename}.{Dota2Small.NEXT_ITEM_INDEX_SUFFIX}"
        nip_index = NextItemIndex(nip_index_file_path)

        if shuffle:
            dataset = NextItemIterableDataset(session_dataset, nip_index)
        else:
            dataset = NextItemDataset(session_dataset, nip_index)

        return dataset

    def _create_token_vocabulary(self):
        token_vocab_file = self.local_path / Dota2Small.TOKEN_VOCABULARY_FILENAME

        if token_vocab_file.exists():
            return

        vocab_builder = VocabularyBuilder()
        vocab_builder.add_token(Dota2Small.PAD_TOKEN_NAME)

        for basename in [Dota2Small.TRAIN_BASENAME, Dota2Small.VALID_BASENAME, Dota2Small.TEST_BASENAME]:
            dataset = self.create_session_dataset(basename, self.delimiter)

            for idx in tqdm(range(len(dataset)), desc=f"Tokenizing items from: {basename}"):
                session = dataset[idx]
                session_tokens = session[ITEM_SEQ_ENTRY_NAME]

                for token in session_tokens:
                    vocab_builder.add_token(token)

        vocabulary = vocab_builder.build()
        with token_vocab_file.open("w") as file:
            CSVVocabularyReaderWriter().write(vocabulary, file)

    def _get_remote_archive_name(self):
        return str(self._get_remote_archive_path()).split("/")[-1]

    def _get_local_archive_path(self) -> Path:
        return self.local_path / self._get_remote_archive_name()

    def _get_remote_archive_path(self) -> Path:
        parsed_remote_uri = urllib.parse.urlparse(self.remote_uri)
        return Path(parsed_remote_uri.path)

    def _remote_uri_is_file(self):
        parsed_remote_uri = urllib.parse.urlparse(self.remote_uri)
        return parsed_remote_uri.scheme == "file"

    def _copy_remote_archive(self):
        if not self.remote_uri:
            return

        ds_local_archive_file_path = self._get_local_archive_path()

        if not ds_local_archive_file_path.exists():
            parsed_remote_uri = urllib.parse.urlparse(self.remote_uri)
            if not self._remote_uri_is_file():
                raise Exception("Only direct file copy is supported right now.")

            print("Copying remote data file")
            if not ds_local_archive_file_path.parent.exists():
                ds_local_archive_file_path.parent.mkdir(parents=True)
            shutil.copyfile(parsed_remote_uri.path, ds_local_archive_file_path)

    def _guess_tarfile_mode(self) -> str:
        archive_name = self._get_remote_archive_name()
        suffix = archive_name.split(".")[-1]

        if suffix == "tar":
            return "r"
        else:
            return f"r:{suffix}"

    def _extract_archive(self):
        archive_path = self._get_remote_archive_path()
        destination = Path(self.local_path)

        print(f"Extracting data from {archive_path} to {destination}")
        archive = tarfile.open(name=archive_path, mode=self._guess_tarfile_mode())
        archive.extractall(destination)




