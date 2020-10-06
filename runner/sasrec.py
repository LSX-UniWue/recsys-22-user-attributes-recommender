from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from data.base.reader import CsvDatasetReader, CsvDatasetIndex
from data.datasets import TARGET_ENTRY_NAME, ITEM_SEQ_ENTRY_NAME
from data.datasets.nextitem import NextItemDataset, NextItemIndex
from data.datasets.posneg import PosNegSessionDataset
from data.datasets.session import ItemSessionDataset, ItemSessionParser
from data.utils import create_indexed_header, read_csv_header
from models.sasrec.sas_rec_model import SASRecModel
from modules import SASRecModule

from padding import padded_session_collate
from tokenization.tokenizer import Tokenizer
from tokenization.vocabulary import CSVVocabularyReaderWriter, Vocabulary, VocabularyReaderWriter

from dependency_injector import containers, providers


def provide_vocabulary(serializer: VocabularyReaderWriter, file: str) -> Vocabulary:
    return serializer.read(Path(file).open("r"))


def provide_tokenizer(vocabulary, special_tokens):
    return Tokenizer(vocabulary, **special_tokens)


def provide_session_dataset(csv_file: str, csv_file_index: str, tokenizer: Tokenizer, delimiter: str, item_column_name: str):
    index = CsvDatasetIndex(Path(csv_file_index))
    reader = CsvDatasetReader(Path(csv_file), index)
    header = create_indexed_header(read_csv_header(Path(csv_file), delimiter=delimiter))
    return ItemSessionDataset(reader, ItemSessionParser(header, item_column_name, delimiter), tokenizer)


def provide_nextitem_dataset(csv_file: str, csv_file_index: str, nip_index: str, tokenizer: Tokenizer, delimiter: str, item_column_name: str):
    index = CsvDatasetIndex(Path(csv_file_index))
    reader = CsvDatasetReader(Path(csv_file), index)
    header = create_indexed_header(read_csv_header(Path(csv_file), delimiter=delimiter))
    session_dataset = ItemSessionDataset(reader, ItemSessionParser(header, item_column_name, delimiter), tokenizer)

    return NextItemDataset(session_dataset, NextItemIndex(Path(nip_index)))


def provide_posneg_dataset(csv_file: str, csv_file_index: str, tokenizer: Tokenizer, delimiter: str, item_column_name: str):
    index = CsvDatasetIndex(Path(csv_file_index))
    reader = CsvDatasetReader(Path(csv_file), index)
    header = create_indexed_header(read_csv_header(Path(csv_file), delimiter=delimiter))
    session_dataset = ItemSessionDataset(reader, ItemSessionParser(header, item_column_name, delimiter), tokenizer)

    return PosNegSessionDataset(session_dataset, tokenizer)


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


class SASRecContainer(containers.DeclarativeContainer):

    config = providers.Configuration()

    # tokenizer
    vocabulary_serializer = providers.Singleton(CSVVocabularyReaderWriter, config.tokenizer.vocabulary.delimiter)
    vocabulary = providers.Singleton(provide_vocabulary, vocabulary_serializer, config.tokenizer.vocabulary.file)
    tokenizer = providers.Singleton(provide_tokenizer, vocabulary, config.tokenizer.special_tokens)

    # model
    model = providers.Singleton(
        SASRecModel,
        config.model.transformer_hidden_size,
        config.model.num_transformer_heads,
        config.model.num_transformer_layers,
        config.model.item_vocab_size,
        config.model.max_seq_length,
        config.model.dropout
    )

    module = providers.Singleton(
        SASRecModule,
        model,
        config.module.batch_size,
        config.module.learning_rate,
        config.module.beta_1,
        config.module.beta_2,
        tokenizer,
        config.module.batch_first,
        config.module.metrics_k
    )

    train_dataset = providers.Factory(
        provide_posneg_dataset,
        config.datasets.train.dataset.csv_file,
        config.datasets.train.dataset.csv_file_index,
        tokenizer,
        config.datasets.train.dataset.delimiter,
        config.datasets.train.dataset.item_column_name
    )

    validation_dataset = providers.Factory(
        provide_nextitem_dataset,
        config.datasets.validation.dataset.csv_file,
        config.datasets.validation.dataset.csv_file_index,
        config.datasets.validation.dataset.nip_index_file,
        tokenizer,
        config.datasets.validation.dataset.delimiter,
        config.datasets.validation.dataset.item_column_name
    )
    
    test_dataset = providers.Factory(
        provide_nextitem_dataset,
        config.datasets.test.dataset.csv_file,
        config.datasets.test.dataset.csv_file_index,
        config.datasets.test.dataset.nip_index_file,
        tokenizer,
        config.datasets.test.dataset.delimiter,
        config.datasets.test.dataset.item_column_name
    )
    
    train_loader = providers.Factory(
        provide_posneg_loader,
        train_dataset,
        config.datasets.train.loader.batch_size,
        config.datasets.train.loader.max_seq_length,
        tokenizer
    )

    validation_loader = providers.Factory(
        provide_nextit_loader,
        validation_dataset,
        config.datasets.validation.loader.batch_size,
        config.datasets.validation.loader.max_seq_length,
        tokenizer
    )

    test_loader = providers.Factory(
        provide_nextit_loader,
        test_dataset,
        config.datasets.test.loader.batch_size,
        config.datasets.test.loader.max_seq_length,
        tokenizer
    )

    checkpoint = providers.Singleton(
        ModelCheckpoint,
        monitor=config.trainer.checkpoint.monitor,
        save_top_k=config.trainer.checkpoint.save_top_k
    )

    trainer = providers.Singleton(
        Trainer,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        default_root_dir=config.trainer.default_root_dir,
        checkpoint_callback=checkpoint
    )


def main():
    container = SASRecContainer()
    container.config.from_yaml("sasrec.yml")
    module = container.module()

    trainer = container.trainer()
    trainer.fit(module, train_dataloader=container.train_loader(), val_dataloaders=container.validation_loader())


if __name__ == "__main__":
    main()


