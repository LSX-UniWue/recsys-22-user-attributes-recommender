from pathlib import Path
from typing import Callable

from dependency_injector import providers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets import ITEM_SEQ_ENTRY_NAME
from data.datasets.nextitem import NextItemDataset, NextItemIndex
from data.datasets.posneg import PosNegSessionDataset
from data.datasets.session import ItemSessionDataset, ItemSessionParser
from data.utils import create_indexed_header, read_csv_header
from metrics.utils.metric_utils import build_metrics
from padding import padded_session_collate
from tokenization.tokenizer import Tokenizer
from tokenization.vocabulary import VocabularyReaderWriter, Vocabulary, CSVVocabularyReaderWriter


def build_posnet_dataset_provider_factory(tokenizer_provider: providers.Provider,
                                          dataset_config: providers.ConfigurationOption
                                          ) -> providers.Factory:

    def provide_posneg_dataset(csv_file: str,
                               csv_file_index: str,
                               nip_index: str,  # unused but necessary to match the call signature
                               tokenizer: Tokenizer,
                               delimiter: str,
                               item_column_name: str):
        index = CsvDatasetIndex(Path(csv_file_index))
        reader = CsvDatasetReader(Path(csv_file), index)
        header = create_indexed_header(read_csv_header(Path(csv_file), delimiter=delimiter))
        session_dataset = ItemSessionDataset(reader, ItemSessionParser(header, item_column_name, delimiter), tokenizer)

        return PosNegSessionDataset(session_dataset, tokenizer)

    return build_dataset_provider_factory(provide_posneg_dataset, tokenizer_provider, dataset_config)


def provide_vocabulary(serializer: VocabularyReaderWriter, file: str) -> Vocabulary:
    return serializer.read(Path(file).open("r"))


def provide_tokenizer(vocabulary, special_tokens):
    return Tokenizer(vocabulary, **special_tokens)


def build_tokenizer_provider(config: providers.Configuration) -> providers.Singleton:
    vocabulary_serializer = providers.Singleton(CSVVocabularyReaderWriter, config.tokenizer.vocabulary.delimiter)
    vocabulary = providers.Singleton(provide_vocabulary, vocabulary_serializer, config.tokenizer.vocabulary.file)
    return providers.Singleton(provide_tokenizer, vocabulary, config.tokenizer.special_tokens)


def build_session_loader_provider_factory(dataset_config: providers.ConfigurationOption,
                                          tokenizer_provider: providers.Provider
                                          ) -> providers.Factory:
    dataset = build_session_dataset_provider_factory(tokenizer_provider, dataset_config)
    return providers.Factory(
        provide_posneg_loader,
        dataset,
        dataset_config.loader.batch_size,
        dataset_config.loader.max_seq_length,
        tokenizer_provider
    )


def build_nextitem_loader_provider_factory(dataset_config: providers.ConfigurationOption,
                                           tokenizer_provider: providers.Provider
                                           ) -> providers.Factory:
    dataset = build_nextitem_dataset_provider_factory(tokenizer_provider, dataset_config)
    return providers.Factory(
        provide_nextit_loader,
        dataset,
        dataset_config.loader.batch_size,
        dataset_config.loader.max_seq_length,
        tokenizer_provider
    )


def build_posneg_loader_provider_factory(dataset_config: providers.ConfigurationOption,
                                         tokenizer_provider: providers.Provider
                                         ) -> providers.Factory:
    dataset = build_posnet_dataset_provider_factory(tokenizer_provider, dataset_config)
    return providers.Factory(
        provide_posneg_loader,
        dataset,
        dataset_config.loader.batch_size,
        dataset_config.loader.max_seq_length,
        tokenizer_provider
    )


def build_session_dataset_provider_factory(tokenizer_provider: providers.Provider,
                                           dataset_config: providers.ConfigurationOption
                                           ) -> providers.Factory:
    def provide_session_dataset(csv_file: str,
                                csv_file_index: str,
                                tokenizer: Tokenizer,
                                delimiter: str,
                                item_column_name: str
                                ):
        index = CsvDatasetIndex(Path(csv_file_index))
        reader = CsvDatasetReader(Path(csv_file), index)
        header = create_indexed_header(read_csv_header(Path(csv_file), delimiter=delimiter))
        return ItemSessionDataset(reader, ItemSessionParser(header, item_column_name, delimiter), tokenizer)

    dataset_config = dataset_config.dataset

    return providers.Factory(
        provide_session_dataset,
        dataset_config.csv_file,
        dataset_config.csv_file_index,
        tokenizer_provider,
        dataset_config.delimiter,
        dataset_config.item_column_name
    )


def build_dataset_provider_factory(dataset_build_fn: Callable[[str, str, str, Tokenizer, str, str], Dataset],
                                   tokenizer_provider: providers.Provider,
                                   dataset_config: providers.ConfigurationOption
                                   ) -> providers.Factory:

    dataset_config = dataset_config.dataset

    return providers.Factory(
        dataset_build_fn,
        dataset_config.csv_file,
        dataset_config.csv_file_index,
        dataset_config.nip_index_file,
        tokenizer_provider,
        dataset_config.delimiter,
        dataset_config.item_column_name
    )


def build_nextitem_dataset_provider_factory(tokenizer_provider: providers.Provider,
                                            dataset_config: providers.ConfigurationOption
                                            ) -> providers.Factory:

    def provide_nextitem_dataset(csv_file: str,
                                 csv_file_index: str,
                                 nip_index: str,
                                 tokenizer: Tokenizer,
                                 delimiter: str,
                                 item_column_name: str
                                 ) -> Dataset:
        index = CsvDatasetIndex(Path(csv_file_index))
        reader = CsvDatasetReader(Path(csv_file), index)
        header = create_indexed_header(read_csv_header(Path(csv_file), delimiter=delimiter))
        session_dataset = ItemSessionDataset(reader, ItemSessionParser(header, item_column_name, delimiter), tokenizer)

        return NextItemDataset(session_dataset, NextItemIndex(Path(nip_index)))

    return build_dataset_provider_factory(provide_nextitem_dataset, tokenizer_provider, dataset_config)


def provide_posneg_loader(dataset: Dataset, batch_size: int, max_seq_length: int, tokenizer: Tokenizer):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=padded_session_collate(
            max_seq_length,
            tokenizer.pad_token_id,
            ["session", "positive_samples", "negative_samples"],
            "session"
        )
    )


def provide_nextit_loader(dataset: Dataset, batch_size: int, max_seq_length: int, tokenizer: Tokenizer):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=padded_session_collate(
            max_seq_length,
            tokenizer.pad_token_id,
            [ITEM_SEQ_ENTRY_NAME],
            ITEM_SEQ_ENTRY_NAME
        )
    )


def build_standard_trainer(config: providers.Configuration) -> providers.Singleton:
    checkpoint = build_standard_model_checkpoint(config)
    return providers.Singleton(
        Trainer,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        default_root_dir=config.trainer.default_root_dir,
        checkpoint_callback=checkpoint,
        gpus=config.trainer.gpus
    )


def build_standard_model_checkpoint(config: providers.Configuration) -> providers.Singleton:
    return providers.Singleton(
        ModelCheckpoint,
        filepath=config.trainer.checkpoint.filepath,
        monitor=config.trainer.checkpoint.monitor,
        save_top_k=config.trainer.checkpoint.save_top_k,
    )


def build_metrics_provider(config: providers.ConfigurationOption
                           ) -> providers.Singleton:
    return providers.Singleton(
        build_metrics,
        config.ids,
        config.k
    )