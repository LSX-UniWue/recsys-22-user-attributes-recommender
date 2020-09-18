import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pathlib import Path

from configs.models.gru.gru_config import GRUConfig
from configs.training.gru.gru_config import GRUTrainingConfig
from data.base.reader import CsvDatasetReader, CsvDatasetIndex
from data.datasets.nextitem import NextItemIndex, NextItemIterableDataset
from data.datasets.session import ItemSessionDataset, ItemSessionParser
from data.mp import mp_worker_init_fn
from data.utils import create_indexed_header, read_csv_header
from metrics.ranking_metrics import RecallAtMetric, MRRAtMetric
from modules.gru_module import GRUModule
from padding import padded_session_collate


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


def add_trainer_args(parser: ArgumentParser):
    trainer_parser = ArgumentParser()
    trainer_parser = pl.Trainer.add_argparse_args(trainer_parser)

    trainer_group = parser.add_argument_group("PL Trainer")

    for action in trainer_parser._actions:
        if isinstance(action, argparse._StoreAction):
            trainer_group.add_argument(
                action.option_strings[0],
                dest=action.dest,
                nargs=action.nargs,
                const=action.const,
                default=action.default,
                type=action.type,
                choices=action.choices,
                help=action.help,
                metavar=action.metavar
            )


def main():
    parser = ArgumentParser()
    parser = GRUConfig.add_model_specific_args(parser)
    parser = GRUTrainingConfig.add_model_specific_args(parser)
    add_trainer_args(parser)

    args = parser.parse_args()

    model_config = GRUConfig.from_args(vars(args))
    training_config = GRUTrainingConfig.from_args(vars(args))

    metrics = [
        RecallAtMetric(k=1),
        RecallAtMetric(k=5),
        MRRAtMetric(k=1),
        MRRAtMetric(k=5)
    ]

    base = Path("/home/dallmann/uni/research/dota/datasets/small/splits")

    train_data_file_path = base / "train.csv"
    train_index_file_path = base / "train.idx"
    train_nip_index_file_path = base / "train.nip.idx"

    valid_data_file_path = base / "valid.csv"
    valid_index_file_path = base / "valid.idx"
    valid_nip_index_file_path = base / "valid.nip.idx"

    test_data_file_path = base / "test.csv"
    test_index_file_path = base / "test.idx"
    test_nip_index_file_path = base / "test.nip.idx"

    delimiter = "\t"

    train_dataset = create_dataset(train_data_file_path, train_index_file_path, train_nip_index_file_path, delimiter)
    training_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        collate_fn=padded_session_collate(model_config.max_seq_length),
        num_workers=1,
        worker_init_fn=mp_worker_init_fn
    )

    valid_dataset = create_dataset(valid_data_file_path, valid_index_file_path, valid_nip_index_file_path, delimiter)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training_config.batch_size,
        collate_fn=padded_session_collate(model_config.max_seq_length),
    )

    checkpoint_path = args.default_root_dir
    checkpoint_callback = ModelCheckpoint(f"{checkpoint_path}", save_last=True, save_top_k=3, save_weights_only=False)

    module = GRUModule(training_config, model_config, metrics=metrics)
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)
    trainer.fit(module, train_dataloader=training_loader, val_dataloaders=valid_loader)

    if False:
        test_dataset = create_dataset(test_data_file_path, test_index_file_path, test_nip_index_file_path, delimiter)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=padded_session_collate(max_seq_length),
        )

        training_config = GRUTrainingConfig.from_json_file(Path(f"{checkpoint_path}/{GRUTrainingConfig.MODEL_CONFIG_CONFIG_FILE}.json"))
        model_config = GRUConfig.from_json_file(Path(f"{checkpoint_path}/{GRUConfig.MODEL_CONFIG_CONFIG_FILE}.json"))

        metrics = [
            RecallAtMetric(k=1),
            RecallAtMetric(k=5),
            MRRAtMetric(k=1),
            MRRAtMetric(k=5)
        ]

        module = GRUModule(training_config=training_config, model_config=model_config, metrics=metrics)
        from pytorch_lightning.utilities.cloud_io import load as pl_load

        checkpoint = pl_load(f"{checkpoint_path}/last.ckpt", map_location=lambda storage, loc: storage)
        module.load_state_dict(checkpoint["state_dict"], strict=False)

        # trainer = pl.Trainer(gpus=None, max_epochs=10, check_val_every_n_epoch=1)
        # trainer.fit(model, train_dataloader=training_loader, val_dataloaders=validation_loader)

        trainer = pl.Trainer(gpus=None, max_epochs=max_epochs, val_check_interval=val_check_interval, limit_test_batches=100, checkpoint_callback=False)

        trainer.test(module, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
