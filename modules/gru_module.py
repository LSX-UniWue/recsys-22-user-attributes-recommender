from typing import Union, List, Dict

import torch

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from models.gru.gru_model import GRUSeqItemRecommenderModel
from modules.util.module_util import get_padding_mask
from tokenization.tokenizer import Tokenizer


def _convert_target_for_multi_label_margin_loss(target: torch.Tensor,
                                                num_classes: int,
                                                pad_token_id: int
                                                ) -> torch.Tensor:
    converted_target = F.pad(target, (0, num_classes - target.size()[1]))
    converted_target[converted_target == pad_token_id] = -1
    return converted_target


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

    def training_step(self, batch, batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        logits = self._forward(input_seq, padding_mask)

        loss_target = target
        if len(target.size()) == 1:
            loss_fnc = nn.CrossEntropyLoss()
        else:
            loss_fnc = nn.MultiLabelMarginLoss()
            loss_target = _convert_target_for_multi_label_margin_loss(target, len(self.tokenizer), self.tokenizer.pad_token_id)
        loss = loss_fnc(logits, loss_target)

        return {
            "loss": loss
        }

    def validation_step(self, batch, batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        logits = self._forward(input_seq, padding_mask)

        mask = None
        loss_target = target
        if len(target.size()) == 1:
            loss_func = nn.CrossEntropyLoss()
        else:
            # first calc the mask
            mask = ~ target.eq(self.tokenizer.pad_token_id)
            loss_func = nn.MultiLabelMarginLoss()

            # after calculating the mask, adapt the target
            loss_target = _convert_target_for_multi_label_margin_loss(target, len(self.tokenizer), self.tokenizer.pad_token_id)
        loss = loss_func(logits, loss_target)
        self.log("val_loss", loss, prog_bar=True)

        for name, metric in self.metrics.items():
            step_value = metric(logits, target, mask=mask)
            self.log(name, step_value, prog_bar=True)

    def validation_epoch_end(self, outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]):
        for name, metric in self.metrics.items():
            self.log(name, metric.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]):
        self.validation_epoch_end(outputs)

    def _forward(self, session, lengths):
        return self.model(session, lengths)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta_1, self.beta_2)
        )
