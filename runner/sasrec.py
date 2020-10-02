from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from configs.models.sasrec.sas_rec_config import SASRecConfig
from configs.training.sasrec.sas_rec_config import SASRecTrainingConfig
from data.base.reader import CsvDatasetReader, CsvDatasetIndex
from data.datasets.posneg import PosNegSessionDataset
from data.datasets.session import ItemSessionDataset, ItemSessionParser
from data.utils import create_indexed_header, read_csv_header
from modules import SASRecModule
import torch

from padding import padded_session_collate
from tokenization.tokenizer import Tokenizer
from tokenization.vocabulary import CSVVocabularyReaderWriter


def main():
    csv_file_path = Path("/home/dallmann/uni/research/dota/datasets/small/splits/train.csv")
    csv_file_index_path = Path("/home/dallmann/uni/research/dota/datasets/small/splits/train.idx")
    vocab_file_path = Path("/home/dallmann/uni/research/dota/datasets/small/splits/items.vocab")

    delimiter = "\t"
    index = CsvDatasetIndex(csv_file_index_path)
    reader = CsvDatasetReader(csv_file_path, index)
    header = create_indexed_header(read_csv_header(csv_file_path, delimiter=delimiter))
    tokenizer = Tokenizer(CSVVocabularyReaderWriter().read(vocab_file_path.open(mode="r")), pad_token="<PAD>")
    session_dataset = ItemSessionDataset(reader, ItemSessionParser(header, "item_id", delimiter), tokenizer)
    dataset = PosNegSessionDataset(session_dataset, tokenizer)

    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=padded_session_collate(2047, 0, ["session", "positive_samples", "negative_samples"], "session"))

    module = SASRecModule(SASRecTrainingConfig(2), SASRecConfig(248, 2047, 16, 1, 1, 0.1), tokenizer, batch_first=True)

    trainer = Trainer()
    trainer.fit(module, train_dataloader=loader, val_dataloaders=loader)



if __name__ == "__main__":
    main()