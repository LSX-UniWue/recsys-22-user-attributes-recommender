from typing import Union, List, Dict, Optional

import torch

import pytorch_lightning as pl
import torch.nn as nn

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from models.gru.gru_model import GRUSeqItemRecommenderModel
from modules.util.module_util import get_padding_mask
from tokenization.tokenizer import Tokenizer


class GRUModule(pl.LightningModule):

    def __init__(self,
                 model: GRUSeqItemRecommenderModel,
                 lr: float,
                 beta_1: float,
                 beta_2: float,
                 tokenizer: Tokenizer,
                 metrics: torch.nn.ModuleDict
                 ):

        super(GRUModule, self).__init__()

        self.model = model
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.loss = nn.CrossEntropyLoss()

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        logits = self._forward(input_seq, padding_mask, batch_idx)
        loss = self.loss(logits, target)

        return {
            "loss": loss
        }

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> None:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        logits = self._forward(input_seq, padding_mask, batch_idx)

        loss = self.loss(logits, target)
        self.log("val_loss", loss, prog_bar=True)

        for name, metric in self.metrics.items():
            step_value = metric(logits, target)
            self.log(name, step_value, prog_bar=True)

    def validation_epoch_end(self,
                             outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
                             ) -> None:
        for name, metric in self.metrics.items():
            self.log(name, metric.compute(), prog_bar=True)

    def test_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int
                  ):
        self.validation_step(batch, batch_idx)

    def test_epoch_end(self,
                       outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
                       ):
        self.validation_epoch_end(outputs)

    def _forward(self,
                 session,
                 lengths,
                 batch_idx
                 ):
        return self.model(session, lengths, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta_1, self.beta_2)
        )
