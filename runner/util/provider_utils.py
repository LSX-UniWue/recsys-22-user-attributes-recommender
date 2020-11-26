import os
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional

from dependency_injector import providers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader

from data.base.reader import CsvDatasetIndex, CsvDatasetReader
from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, \
    NEGATIVE_SAMPLES_ENTRY_NAME
from data.datasets.nextitem import NextItemDataset
from data.datasets.index import SessionPositionIndex
from data.datasets.prepare import Processor, build_processors, PositiveNegativeSampler
from data.datasets.session import ItemSessionDataset, ItemSessionParser, PlainSessionDataset
from data.mp import mp_worker_init_fn
from data.utils import create_indexed_header, read_csv_header
from logger.GradientLoggerCallback import GradientLoggerCallback
from logger.MetricLoggerCallback import MetricLoggerCallback
from logger.TrainLossLoggerCallback import TrainLossLoggerCallback
from metrics.utils.metric_utils import build_metrics
from data.collate import padded_session_collate
from tokenization.tokenizer import Tokenizer
from tokenization.vocabulary import VocabularyReaderWriter, Vocabulary, CSVVocabularyReaderWriter


def build_session_parser(csv_file: Path,
                         item_column_name: str,
                         delimiter: str,
                         additional_features: Dict[str, Any],
                         item_separator: str
                         ) -> ItemSessionParser:
    header = create_indexed_header(read_csv_header(csv_file, delimiter=delimiter))
    return ItemSessionParser(header, item_column_name,
                             additional_features=additional_features,
                             item_separator=item_separator,
                             delimiter=delimiter)


def build_processors_provider(dataset_config: providers.ConfigurationOption,
                              additional_objects: Dict[str, Any]
                              ) -> providers.Factory:
    return providers.Factory(
        build_processors,
        dataset_config,
        **additional_objects
    )


def build_posnet_dataset_provider_factory(tokenizer_provider: providers.Provider,
                                          processors_provider: providers.Provider,
                                          dataset_config: providers.ConfigurationOption
                                          ) -> providers.Factory:
    def provide_posneg_dataset(csv_file: str,
                               csv_file_index: str,
                               nip_index: str,  # unused but necessary to match the call signature
                               processors: List[Processor],
                               tokenizer: Tokenizer,
                               delimiter: str,
                               item_column_name: str,
                               item_separator: str,
                               additional_features: Dict[str, Any]
                               ) -> Dataset:
        index = CsvDatasetIndex(Path(csv_file_index))
        csv_file = Path(csv_file)
        reader = CsvDatasetReader(csv_file, index)

        session_parser = build_session_parser(csv_file=csv_file,
                                              item_column_name=item_column_name,
                                              delimiter=delimiter,
                                              item_separator=item_separator,
                                              additional_features=additional_features)
        basic_dataset = PlainSessionDataset(reader, session_parser)

        has_pos_neg_sampler_processor = False
        for processor in processors:
            if isinstance(processor, PositiveNegativeSampler):
                has_pos_neg_sampler_processor = True
        if not has_pos_neg_sampler_processor:
            raise ValueError('please configure a pos neg sampler')
        return ItemSessionDataset(basic_dataset, processors)

    return build_dataset_provider_factory(provide_posneg_dataset, tokenizer_provider, processors_provider,
                                          dataset_config)


def provide_vocabulary(serializer: VocabularyReaderWriter, file: str) -> Vocabulary:
    return serializer.read(Path(file).open("r"))


def provide_tokenizer(vocabulary, special_tokens):
    return Tokenizer(vocabulary, **special_tokens)


def build_tokenizer_provider(config: providers.Configuration) -> providers.Singleton:
    vocabulary_serializer = providers.Singleton(CSVVocabularyReaderWriter, config.tokenizer.vocabulary.delimiter)
    vocabulary = providers.Singleton(provide_vocabulary, vocabulary_serializer, config.tokenizer.vocabulary.file)
    return providers.Singleton(provide_tokenizer, vocabulary, config.tokenizer.special_tokens)


def build_session_loader_provider_factory(dataset_config: providers.ConfigurationOption,
                                          tokenizer_provider: providers.Provider,
                                          processors_providers: providers.Provider
                                          ) -> providers.Factory:
    dataset = build_session_dataset_provider_factory(tokenizer_provider, processors_providers, dataset_config)
    dataset_loader_config = dataset_config.loader
    return providers.Factory(
        provide_session_loader,
        dataset,
        dataset_loader_config.batch_size,
        dataset_loader_config.max_seq_length,
        dataset_loader_config.max_seq_step_length,
        dataset_loader_config.num_workers,
        tokenizer_provider
    )


def build_nextitem_loader_provider_factory(dataset_config: providers.ConfigurationOption,
                                           tokenizer_provider: providers.Provider,
                                           processors_provider: providers.Provider
                                           ) -> providers.Factory:
    dataset = build_nextitem_dataset_provider_factory(tokenizer_provider, processors_provider, dataset_config)
    return providers.Factory(
        provide_nextit_loader,
        dataset,
        dataset_config.loader.batch_size,
        dataset_config.loader.max_seq_length,
        dataset_config.loader.max_seq_step_length,
        dataset_config.loader.shuffle,
        dataset_config.loader.num_workers,
        tokenizer_provider
    )


def build_posneg_loader_provider_factory(dataset_config: providers.ConfigurationOption,
                                         tokenizer_provider: providers.Provider,
                                         processor_provider_provider: providers.Provider
                                         ) -> providers.Factory:
    dataset = build_posnet_dataset_provider_factory(tokenizer_provider, processor_provider_provider, dataset_config)
    dataset_loader_config = dataset_config.loader
    return providers.Factory(
        provide_session_loader,
        dataset,
        dataset_loader_config.batch_size,
        dataset_loader_config.max_seq_length,
        dataset_loader_config.max_seq_step_length,
        dataset_loader_config.num_workers,
        tokenizer_provider
    )


def build_session_dataset_provider_factory(tokenizer_provider: providers.Provider,
                                           processors_provider: providers.Provider,
                                           dataset_config: providers.ConfigurationOption
                                           ) -> providers.Factory:
    def provide_session_dataset(csv_file: str,
                                csv_file_index: str,
                                processors: List[Processor],
                                tokenizer: Tokenizer,
                                delimiter: str,
                                item_column_name: str,
                                item_separator: str,
                                additional_features: Dict[str, Any],
                                truncated_index_path: str
                                ) -> Dataset:
        index = CsvDatasetIndex(Path(csv_file_index))
        csv_file = Path(csv_file)
        reader = CsvDatasetReader(csv_file, index)
        session_parser = build_session_parser(csv_file=csv_file,
                                              item_column_name=item_column_name,
                                              delimiter=delimiter,
                                              item_separator=item_separator,
                                              additional_features=additional_features)

        basic_dataset = PlainSessionDataset(reader, session_parser)
        if truncated_index_path is not None:
            index = SessionPositionIndex(Path(truncated_index_path))
            return NextItemDataset(basic_dataset,
                                   index=index,
                                   processors=processors,
                                   add_target=False)

        return ItemSessionDataset(basic_dataset, processors=processors)

    dataset_config = dataset_config.dataset
    parser_config = dataset_config.parser

    return providers.Factory(
        provide_session_dataset,
        dataset_config.csv_file,
        dataset_config.csv_file_index,
        processors_provider,
        tokenizer_provider,
        parser_config.delimiter,
        parser_config.item_column_name,
        parser_config.item_separator,
        parser_config.additional_features,
        dataset_config.truncated_seq_index_file
    )


def build_dataset_provider_factory(
        dataset_build_fn: Callable[[str, str, str, List[Processor], Tokenizer, str, str, str, Dict[str, Any]], Dataset],
        tokenizer_provider: providers.Provider,
        processors_provider: providers.Provider,
        dataset_config: providers.ConfigurationOption
        ) -> providers.Factory:
    dataset_config = dataset_config.dataset
    parser_config = dataset_config.parser

    return providers.Factory(
        dataset_build_fn,
        dataset_config.csv_file,
        dataset_config.csv_file_index,
        dataset_config.nip_index_file,
        processors_provider,
        tokenizer_provider,
        parser_config.delimiter,
        parser_config.item_column_name,
        parser_config.item_separator,
        parser_config.additional_features
    )


def build_nextitem_dataset_provider_factory(tokenizer_provider: providers.Provider,
                                            preprocessor_provider: providers.Provider,
                                            dataset_config: providers.ConfigurationOption
                                            ) -> providers.Factory:
    def provide_nextitem_dataset(csv_file: str,
                                 csv_file_index: str,
                                 nip_index: str,
                                 preprocessors: List[Processor],
                                 tokenizer: Tokenizer,
                                 delimiter: str,
                                 item_column_name: str,
                                 item_separator: str,
                                 additional_features: Dict[str, Any]
                                 ) -> Dataset:
        index = CsvDatasetIndex(Path(csv_file_index))
        csv_file = Path(csv_file)
        reader = CsvDatasetReader(csv_file, index)
        session_parser = build_session_parser(csv_file=csv_file,
                                              item_column_name=item_column_name,
                                              delimiter=delimiter,
                                              item_separator=item_separator,
                                              additional_features=additional_features)
        basic_dataset = PlainSessionDataset(reader, session_parser)
        return NextItemDataset(basic_dataset, SessionPositionIndex(Path(nip_index)), processors=preprocessors)

    return build_dataset_provider_factory(provide_nextitem_dataset, tokenizer_provider, preprocessor_provider,
                                          dataset_config)


def _build_entries_to_pad(max_seq_length: int,
                          max_seq_step_length: Optional[int]
                          ) -> Dict[str, List[int]]:
    entries_to_pad = {
        ITEM_SEQ_ENTRY_NAME: [max_seq_length],
        POSITIVE_SAMPLES_ENTRY_NAME: [max_seq_length],
        NEGATIVE_SAMPLES_ENTRY_NAME: [max_seq_length]
    }

    if max_seq_step_length is not None:
        for key in [ITEM_SEQ_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME]:
            entries_to_pad[key].append(max_seq_step_length)
        entries_to_pad[TARGET_ENTRY_NAME] = [max_seq_step_length]

    return entries_to_pad


def provide_session_loader(dataset: Dataset,
                           batch_size: int,
                           max_seq_length: int,
                           max_seq_step_length: int,
                           num_workers: int,
                           tokenizer: Tokenizer
                           ) -> DataLoader:
    init_worker_fn = None if num_workers == 0 else mp_worker_init_fn

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=padded_session_collate(
            pad_token_id=tokenizer.pad_token_id,
            entries_to_pad=_build_entries_to_pad(max_seq_length, max_seq_step_length),
            session_length_entry="session"
        ),
        num_workers=num_workers,
        worker_init_fn=init_worker_fn
    )


def provide_nextit_loader(dataset: Dataset,
                          batch_size: int,
                          max_seq_length: int,
                          max_seq_step_length: int,
                          shuffle: bool,
                          num_workers: int,
                          tokenizer: Tokenizer
                          ) -> DataLoader:
    init_worker_fn = None if num_workers == 0 else mp_worker_init_fn
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=padded_session_collate(
            pad_token_id=tokenizer.pad_token_id,
            entries_to_pad=_build_entries_to_pad(max_seq_length, max_seq_step_length),
            session_length_entry=ITEM_SEQ_ENTRY_NAME
        ),
        num_workers=num_workers,
        worker_init_fn=init_worker_fn
    )


def build_standard_trainer(config: providers.Configuration) -> providers.Singleton:
    checkpoint = build_standard_model_checkpoint(config)
    logger = build_standard_tensorboard_logger_provider(config)
    logging_callbacks = build_standard_logging_callbacks_provider()

    trainer_config = config.trainer
    return providers.Singleton(
        Trainer,
        limit_train_batches=trainer_config.limit_train_batches,
        limit_val_batches=trainer_config.limit_val_batches,
        default_root_dir=trainer_config.default_root_dir,
        checkpoint_callback=checkpoint,
        gradient_clip_val=trainer_config.gradient_clip_val,
        gpus=trainer_config.gpus,
        max_epochs=trainer_config.max_epochs,
        weights_summary='full',
        logger=logger,
        callbacks=logging_callbacks
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
        config
    )


def build_standard_tensorboard_logger_provider(config: providers.Configuration) -> providers.Singleton:
    log_dir = providers.Singleton(Path,
                                  config.trainer.default_root_dir,
                                  "logs")
    return providers.Singleton(
        TensorBoardLogger,
        save_dir=log_dir,
        name=config.trainer.experiment_name
    )


def build_standard_logging_callbacks_provider() -> providers.List:
    return providers.List(
        MetricLoggerCallback(),
        GradientLoggerCallback(),
        TrainLossLoggerCallback())
