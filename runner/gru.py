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
from dm.dota.small import Dota2Small
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


def train(args):
    model_config = GRUConfig.from_args(vars(args))
    training_config = GRUTrainingConfig.from_args(vars(args))

    metrics = [
        RecallAtMetric(k=1),
        RecallAtMetric(k=5),
        MRRAtMetric(k=1),
        MRRAtMetric(k=5)
    ]

    base = Path("/home/dallmann/uni/research/dota/datasets/small/splits")
    delimiter = "\t"
    dm = Dota2Small(base, delimiter=delimiter)

    checkpoint_path = args.default_root_dir
    checkpoint_callback = ModelCheckpoint(f"{checkpoint_path}", save_last=True, save_top_k=3, save_weights_only=False)

    module = GRUModule(training_config, model_config, metrics=metrics)
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)
    trainer.fit(module, datamodule=dm)


def test(args):
    checkpoint_path = "/tmp/gru-test"
    delimiter = "\t"
    base = Path("/home/dallmann/uni/research/dota/datasets/small/splits")

    training_config = GRUTrainingConfig.from_json_file(Path(f"{checkpoint_path}/{GRUTrainingConfig.MODEL_CONFIG_CONFIG_FILE}.json"))
    model_config = GRUConfig.from_json_file(Path(f"{checkpoint_path}/{GRUConfig.MODEL_CONFIG_CONFIG_FILE}.json"))
    dm = Dota2Small(base, delimiter=delimiter)

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

    trainer = pl.Trainer(limit_test_batches=100, checkpoint_callback=False)
    trainer.test(module, datamodule=dm)


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)
    test_parser = subparsers.add_parser("test")
    test_parser.set_defaults(func=test)

    train_parser = GRUConfig.add_model_specific_args(train_parser)
    train_parser = GRUTrainingConfig.add_model_specific_args(train_parser)
    add_trainer_args(train_parser)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
    # For now we need to specify this to train: train --max_seq_length 2047 --item_voc_size 1032 --default_root_dir /tmp/gru-test --batch_size 256 --val_check_interval 10 --max_epochs 10