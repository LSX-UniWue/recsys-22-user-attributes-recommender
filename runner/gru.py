import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path

from configs.models.gru.gru_config import GRUConfig
from configs.training.gru.gru_config import GRUTrainingConfig
from data.base.reader import CsvSessionDatasetReader, CsvSessionDatasetIndex
from data.datasets.nextitem import NextItemIndex, NextItemIterableDataset
from data.datasets.session import ItemSessionDataset, ItemSessionParser
from data.mp import mp_worker_init_fn
from data.utils import create_indexed_header, read_csv_header
from modules.gru_module import GRUModule
from padding import padded_session_collate


def main():
    torch.set_num_threads(4)
    max_seq_length = 2047
    num_items = 1032
    batch_size = 256

    base = Path("/home/dallmann/uni/research/dota/datasets/small")
    data_file_path = base / "small.csv"
    index_file_path = base / "small.idx"
    next_item_index_file_path = base / "small_nip.idx"
    delimiter = "\t"

    reader = CsvSessionDatasetReader(data_file_path, CsvSessionDatasetIndex(index_file_path))

    dataset = ItemSessionDataset(
        reader,
        ItemSessionParser(
            create_indexed_header(
                read_csv_header(data_file_path, delimiter=delimiter)
            ),
            "item_id", delimiter=delimiter
        )
    )

    seq_item_index = NextItemIndex(next_item_index_file_path)
    seq_item_dataset = NextItemIterableDataset(dataset, seq_item_index, seed=93938293)

    training_loader = DataLoader(
        seq_item_dataset,
        batch_size=batch_size,
        collate_fn=padded_session_collate(max_seq_length),
        num_workers=4,
        worker_init_fn=mp_worker_init_fn
     )

    training_config = GRUTrainingConfig(batch_size=batch_size)
    model_config = GRUConfig(
        item_voc_size=num_items,
        max_seq_length=max_seq_length,
        gru_hidden_size=64,
        gru_token_embedding_size=16,
        gru_num_layers=1
    )

    module = GRUModule(training_config, model_config)
    # trainer = pl.Trainer(gpus=None, max_epochs=10, check_val_every_n_epoch=1)
    # trainer.fit(model, train_dataloader=training_loader, val_dataloaders=validation_loader)

    trainer = pl.Trainer(gpus=None, max_epochs=10)
    trainer.fit(module, train_dataloader=training_loader)


if __name__ == "__main__":
    main()