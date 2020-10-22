from typing import Union, Dict, List

import pytorch_lightning as pl
import torch

from data.datasets import ITEM_SEQ_ENTRY_NAME, USER_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME
from metrics.utils.metric_utils import build_metrics
from models.caser.caser_model import CaserModel


class CaserModule(pl.LightningModule):

    def __init__(self,
                 model: CaserModel,
                 learning_rate: float,
                 weight_decay: float,
                 ks: List[int]
                 ):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.metrics = build_metrics(ks)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        users = batch[USER_ENTRY_NAME] if USER_ENTRY_NAME in batch else None
        pos_items = batch[POSITIVE_SAMPLES_ENTRY_NAME]
        neg_items = batch[NEGATIVE_SAMPLES_ENTRY_NAME]

        pos_logits, neg_logits = self.model(input_seq, users, pos_items, neg_items)

        # TODO: check: this could be the sas loss
        positive_loss = - torch.mean(torch.log(torch.sigmoid(pos_logits)))
        negative_loss = - torch.mean(torch.log(1 - torch.sigmoid(neg_logits)))
        loss = positive_loss + negative_loss

        return {
            'loss': loss
        }

    # FIXME: copy paste code from sas rec module
    def validation_epoch_end(self, outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]) -> None:
        for name, metric in self.metrics.items():
            self.log(name, metric.compute(), prog_bar=True)