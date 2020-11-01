from typing import Union, Dict, List

import pytorch_lightning as pl
import torch

from data.datasets import ITEM_SEQ_ENTRY_NAME, USER_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, \
    NEGATIVE_SAMPLES_ENTRY_NAME, TARGET_ENTRY_NAME
from models.caser.caser_model import CaserModel
from tokenization.tokenizer import Tokenizer


class CaserModule(pl.LightningModule):

    @staticmethod
    def get_users_from_batch(batch):
        return batch[USER_ENTRY_NAME] if USER_ENTRY_NAME in batch else None

    def __init__(self,
                 model: CaserModel,
                 tokenizer: Tokenizer,
                 learning_rate: float,
                 weight_decay: float,
                 metrics: torch.nn.ModuleDict
                 ):
        super().__init__()

        self.model = model

        self.tokenizer = tokenizer

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.metrics = metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def training_step(self, batch, batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        users = self.get_users_from_batch(batch)
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

    # FIXME: a lot of copy paste code
    def validation_step(self, batch, batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        users = self.get_users_from_batch(batch)
        targets = batch[TARGET_ENTRY_NAME]

        batch_size = input_seq.size()[0]

        # provide items that the target item will be ranked against
        # TODO (AD) refactor this into a composable class to allow different strategies for item selection
        device = input_seq.device
        items_to_rank = torch.as_tensor(self.tokenizer.get_vocabulary().ids(), dtype=torch.long, device=device)
        items_to_rank = items_to_rank.repeat([batch_size, 1])

        prediction = self.model(input_seq, users, items_to_rank)

        for name, metric in self.metrics.items():
            step_value = metric(prediction, targets)
            self.log(name, step_value, prog_bar=True)

    # FIXME: copy paste code from sas rec module
    def validation_epoch_end(self, outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]) -> None:
        for name, metric in self.metrics.items():
            self.log(name, metric.compute(), prog_bar=True)