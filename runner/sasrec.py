from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

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
from tokenization.vocabulary import CSVVocabularyReaderWriter, Vocabulary

from dependency_injector import containers, providers



def get_validation_dataset(csv_file_path: Path, csv_file_index_path: Path, nip_index_file_path: Path, vocab_file_path: Path, delimiter: str = "\t"):
    index = CsvDatasetIndex(csv_file_index_path)
    reader = CsvDatasetReader(csv_file_path, index)
    header = create_indexed_header(read_csv_header(csv_file_path, delimiter=delimiter))
    tokenizer = Tokenizer(CSVVocabularyReaderWriter().read(vocab_file_path.open(mode="r")), pad_token="<PAD>")
    session_dataset = ItemSessionDataset(reader, ItemSessionParser(header, "item_id", delimiter), tokenizer)

    dataset = NextItemDataset(session_dataset, NextItemIndex(nip_index_file_path))
    return dataset


def main():
    csv_file_path = Path("/home/dallmann/uni/research/dota/datasets/small/splits/train.csv")
    csv_file_index_path = Path("/home/dallmann/uni/research/dota/datasets/small/splits/train.idx")
    vocab_file_path = Path("/home/dallmann/uni/research/dota/datasets/small/splits/items.vocab")

    validation_csv_file_path = Path("/home/dallmann/uni/research/dota/datasets/small/splits/valid.csv")
    validation_csv_file_index_path = Path("/home/dallmann/uni/research/dota/datasets/small/splits/valid.idx")
    validation_nip_index_path = Path("/home/dallmann/uni/research/dota/datasets/small/splits/valid.nip.idx")

    delimiter = "\t"
    index = CsvDatasetIndex(csv_file_index_path)
    reader = CsvDatasetReader(csv_file_path, index)
    header = create_indexed_header(read_csv_header(csv_file_path, delimiter=delimiter))
    tokenizer = Tokenizer(CSVVocabularyReaderWriter().read(vocab_file_path.open(mode="r")), pad_token="<PAD>")
    session_dataset = ItemSessionDataset(reader, ItemSessionParser(header, "item_id", delimiter), tokenizer)
    dataset = PosNegSessionDataset(session_dataset, tokenizer)

    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=padded_session_collate(2047, 0, ["session", "positive_samples", "negative_samples"], "session"))

    validation_dataset = get_validation_dataset(validation_csv_file_path, validation_csv_file_index_path, validation_nip_index_path, vocab_file_path, delimiter)
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=False, collate_fn=padded_session_collate(2047, 0, [ITEM_SEQ_ENTRY_NAME], ITEM_SEQ_ENTRY_NAME))

    model = SASRecModel(16, 1, 1, 248, 2047, 0.1)
    module = SASRecModule(model, 4, 0.001, 0.99, 0.998, tokenizer, batch_first=True, metrics_k=[5])

    trainer = Trainer(limit_train_batches=10, limit_val_batches=10)
    trainer.fit(module, train_dataloader=train_loader, val_dataloaders=validation_loader)


if __name__ == "__main__":
    main()
