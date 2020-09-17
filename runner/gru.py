import torch
import pytorch_lightning as pl
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


def create_dataset(data_file_path: Path, index_file_path: Path, nip_index_file_path: Path, delimiter):
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


def main():
    torch.set_num_threads(4)
    max_seq_length = 2047
    num_items = 1032
    batch_size = 256
    val_check_interval = 10
    delimiter = "\t"
    max_epochs = 10

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

    mode = "test"

    if mode == "train":

        train_dataset = create_dataset(train_data_file_path, train_index_file_path, train_nip_index_file_path, delimiter)
        training_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=padded_session_collate(max_seq_length),
            num_workers=1,
            worker_init_fn=mp_worker_init_fn
         )

        valid_dataset = create_dataset(valid_data_file_path, valid_index_file_path, valid_nip_index_file_path, delimiter)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            collate_fn=padded_session_collate(max_seq_length),
        )

        training_config = GRUTrainingConfig(batch_size=batch_size)

        model_config = GRUConfig(
            item_voc_size=num_items,
            max_seq_length=max_seq_length,
            gru_hidden_size=64,
            gru_token_embedding_size=16,
            gru_num_layers=1
        )

        metrics = [
            RecallAtMetric(k=1),
            RecallAtMetric(k=5),
            MRRAtMetric(k=1),
            MRRAtMetric(k=5)
        ]

        module = GRUModule(training_config, model_config, metrics)
        # trainer = pl.Trainer(gpus=None, max_epochs=10, check_val_every_n_epoch=1)
        # trainer.fit(model, train_dataloader=training_loader, val_dataloaders=validation_loader)

        trainer = pl.Trainer(gpus=None, max_epochs=max_epochs, val_check_interval=val_check_interval, limit_val_batches=50, limit_train_batches=50, checkpoint_callback=True)
        trainer.fit(module, train_dataloader=training_loader, val_dataloaders=valid_loader)

    else:
        test_dataset = create_dataset(test_data_file_path, test_index_file_path, test_nip_index_file_path, delimiter)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=padded_session_collate(max_seq_length),
        )

        training_config = GRUTrainingConfig(batch_size=batch_size)

        model_config = GRUConfig(
            item_voc_size=num_items,
            max_seq_length=max_seq_length,
            gru_hidden_size=64,
            gru_token_embedding_size=16,
            gru_num_layers=1
        )

        metrics = [
            RecallAtMetric(k=1),
            RecallAtMetric(k=5),
            MRRAtMetric(k=1),
            MRRAtMetric(k=5)
        ]

        module = GRUModule.load_from_checkpoint("/home/dallmann/uni/research/repositories/recommender/runner/lightning_logs/version_0/checkpoints/epoch=9.ckpt", training_config, model_config, metrics)
        # trainer = pl.Trainer(gpus=None, max_epochs=10, check_val_every_n_epoch=1)
        # trainer.fit(model, train_dataloader=training_loader, val_dataloaders=validation_loader)

        trainer = pl.Trainer(gpus=None, max_epochs=max_epochs, val_check_interval=val_check_interval, limit_test_batches=100, checkpoint_callback=False)

        trainer.test(module, test_dataloaders=test_loader)


if __name__ == "__main__":
    main()
