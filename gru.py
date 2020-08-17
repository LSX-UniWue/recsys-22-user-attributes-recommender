import pytorch_lightning as pl
import torch
import torch.nn as nn

from pathlib import Path

from torch.utils.data import DataLoader

from datasets.dataset import SessionDataset, PartialSessionDataset, NegSamplingPurchaseLogDataset, \
    NextItemPredSessionDataset, NextItemPredSessionIndex, NextItemPredIterableDataset, \
    next_item_pred_iterable_dataset_initfn
from padding import padded_session_collate


class GRUSeqItemRecommender(pl.LightningModule):

    def __init__(self, num_items: int, item_embedding_dim: int, hidden_size: int, num_layers: int = 1):
        super(GRUSeqItemRecommender, self).__init__()
        self.item_embeddings = nn.Embedding(num_items, embedding_dim=item_embedding_dim)
        self.gru = nn.GRU(item_embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, num_items, bias=True)

    def forward(self, session, lengths, batch_idx):
        embedded_session = self.item_embeddings(session)
        packed_embedded_session = nn.utils.rnn.pack_padded_sequence(embedded_session, lengths, batch_first=True, enforce_sorted=False)
        _, final_state = self.gru(packed_embedded_session)

        output = self.fcn(final_state)
        return torch.squeeze(output, dim=0)

    def training_step(self, batch, batch_idx):
        prediction = self.forward(batch["session"], batch["session_lengths"], batch_idx)
        loss = nn.CrossEntropyLoss()(prediction, batch["target"])

        return {
            "loss": loss
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def main():
    torch.set_num_threads(4)
    max_seq_length = 2047
    num_items = 247
    seed = 98393939
    base = Path("/home/dallmann/uni/research/dota/datasets/small")
    ds = SessionDataset(base / "small.csv", base / "small.idx")

    #training_index = NextItemPredSessionIndex(ds, base / "small_nip.idx", min_session_length=2)
    training_ds = NextItemPredIterableDataset(ds, 0, len(ds), seed)

    training_loader = DataLoader(training_ds, batch_size=128, collate_fn=padded_session_collate(max_seq_length), worker_init_fn=next_item_pred_iterable_dataset_initfn, num_workers=2)

    model = GRUSeqItemRecommender(num_items, 64, 64, 1)
    # trainer = pl.Trainer(gpus=None, max_epochs=10, check_val_every_n_epoch=1)
    # trainer.fit(model, train_dataloader=training_loader, val_dataloaders=validation_loader)

    trainer = pl.Trainer(gpus=None, max_epochs=10)
    trainer.fit(model, train_dataloader=training_loader)


if __name__ == "__main__":
    main()