import math
from pathlib import Path

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import init
from torch.nn.parameter import Parameter

from torch.utils.data import DataLoader

#TODO regularization
from datasets.dataset import SessionDataset, NegSamplingPurchaseLogDataset, PartialSessionDataset

# from: https://stackoverflow.com/questions/55041080/how-does-pytorch-dataloader-handle-variable-size-data

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([t.shape[0] for t in batch ])
    ## padd
    batch = [torch.Tensor(t) for t in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    mask = (batch != 0)
    return batch, lengths, mask

class BPRModule(pl.LightningModule):

    def __init__(self, dim_U: int, dim_I: int, k: int):
        super(BPRModule, self).__init__()

        self.W = Parameter(torch.zeros([dim_U, k]))
        self.H = Parameter(torch.zeros([dim_I, k]))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform(self.W, a=math.sqrt(5))
        init.kaiming_uniform(self.H, a=math.sqrt(5))

    def forward(self, u, i, j):
        u_emb = self.W.index_select(dim=0, index=u)
        i_emb = self.H.index_select(dim=0, index=i)
        j_emb = self.H.index_select(dim=0, index=j)

        x_uij = (u_emb * i_emb).sum(-1) - (u_emb * j_emb).sum(-1)

        loss = - F.logsigmoid(x_uij).mean()

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        u = batch["user"]
        i = batch["pos"]
        j = batch["neg"]
        return self.forward(u, i, j)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


class BPRMFModule2(pl.LightningModule):

    def __init__(self, dim_U: int, dim_I: int, k: int, reg_lambda: float = 0.0):
        """

        :param dim_U: number of distinct users
        :param dim_I: number of distinct items
        :param k: embedding size for users u and items i.
        """
        super(BPRMFModule2, self).__init__()

        self.W = torch.nn.Embedding(num_embeddings=dim_U, embedding_dim=k)
        self.H = torch.nn.Embedding(num_embeddings=dim_I, embedding_dim=k)
        self.reg_lambda = reg_lambda
        self.reg_loss = torch.nn.L1Loss()

    def forward(self, u, i, j):
        emb_u = self.W(u)
        return self.forward_validate(emb_u, i, j)

    def forward_validate(self, emb_u, i, j):
        emb_i = self.H(i)
        emb_j = self.H(j)

        x_ui = (emb_u * emb_i).sum(-1)
        x_uj = (emb_u * emb_j).sum(-1)

        return x_ui, x_uj

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "loss")

    def validation_step(self, batch, batch_idx):
        # we can't just validate against new data, since we don't know the user, need to build a "user" embedding
        # from the session first
        session = batch["session"]
        emb_u = self.H(session).mean(dim=-2)
        i = batch["pos"]
        j = batch["neg"]

        x_ui, x_uj = self.forward_validate(emb_u, i, j)

        reg = 0
        reg += self.reg_loss(x_ui, target=torch.zeros_like(x_ui))
        reg += self.reg_loss(x_uj, target=torch.zeros_like(x_uj))

        x_uij = x_ui - x_uj

        loss = - F.logsigmoid(x_uij).mean() + self.reg_lambda * reg

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'progress_bar': {'val_loss': avg_loss}, 'log': {'val_loss': avg_loss}, 'val_loss': avg_loss}


    def _shared_step(self, batch, batch_idx, loss_name: str):
        u = batch["user"]
        i = batch["pos"]
        j = batch["neg"]
        x_ui, x_uj = self.forward(u, i, j)

        reg = 0
        reg += self.reg_loss(x_ui, target=torch.zeros_like(x_ui))
        reg += self.reg_loss(x_uj, target=torch.zeros_like(x_uj))

        x_uij = x_ui - x_uj

        loss = - F.logsigmoid(x_uij).mean() + self.reg_lambda * reg

        return {loss_name: loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())




def main():
    max_seq_length = 2047
    seed = 98393939
    base = Path("/home/dallmann/uni/research/dota/datasets/small")
    ds = SessionDataset(base / "small.csv", base / "small.idx")
    #training_ds = RestrictedPurchaseLogDatasetWrapper(ds, start=0, stop=1000)
    training_ds = PartialSessionDataset(ds, start=0, stop=5880238)
    #validation_ds = RestrictedPurchaseLogDatasetWrapper(ds, start=1000, stop=1500)
    validation_ds = PartialSessionDataset(ds, start=5880238, stop=5890238)

    neg_sampling_training_ds = NegSamplingPurchaseLogDataset(training_ds, seed=seed)
    neg_sampling_validation_ds = NegSamplingPurchaseLogDataset(validation_ds, seed=seed)

    # we use the whole user space so that validation can run
    #model = BPRMFModule2(len(training_ds), len(ds.item_id_mapper), 64, 0.1)
    #model = BPRMFModule2(len(training_ds), len(ds.item_id_mapper), 64, 0.1)
    model = BPRModule(len(training_ds), len(ds.item_id_mapper), 64)
    training_loader = DataLoader(neg_sampling_training_ds, batch_size=32)
    validation_loader = DataLoader(neg_sampling_validation_ds, batch_size=16, collate_fn=my_collate)

    #trainer = pl.Trainer(gpus=None, max_epochs=10)
    trainer = pl.Trainer(gpus=None, max_epochs=10, check_val_every_n_epoch=1)
    #trainer.fit(model, train_dataloader=training_loader, val_dataloaders=validation_loader)
    trainer.fit(model, train_dataloader=training_loader)


if __name__ == "__main__":
    main()