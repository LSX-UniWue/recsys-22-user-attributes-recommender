from typing import Union, List, Dict, Optional

import torch

import pytorch_lightning as pl
import torch.nn as nn

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from models.gru.gru_model import GRUSeqItemRecommenderModel
from modules import LOG_KEY_VALIDATION_LOSS
from modules.util.module_util import get_padding_mask, convert_target_to_multi_hot, build_eval_step_return_dict
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

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        logits = self._forward(input_seq, padding_mask)
        loss = self._calc_loss(logits, target)

        return {
            "loss": loss
        }

    # FIXME: same loss as in NARM module
    def _calc_loss(self,
                   logits: torch.Tensor,
                   target: torch.Tensor
                   ) -> float:
        if len(target.size()) == 1:
            # only one item per sequence step
            loss_fnc = nn.CrossEntropyLoss()
            return loss_fnc(logits, target)

        loss_fnc = nn.BCEWithLogitsLoss()
        target = convert_target_to_multi_hot(target, len(self.tokenizer), self.tokenizer.pad_token_id)
        return loss_fnc(logits, target)

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        logits = self._forward(input_seq, padding_mask)

        loss = self._calc_loss(logits, target)
        self.log(LOG_KEY_VALIDATION_LOSS, loss, prog_bar=True)

        mask = None if len(target.size()) == 1 else ~ target.eq(self.tokenizer.pad_token_id)

        for name, metric in self.metrics.items():
            step_value = metric(logits, target, mask=mask)
            self.log(name, step_value, prog_bar=True)

        return build_eval_step_return_dict(logits, target, mask=mask)

    def validation_epoch_end(self,
                             outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
                             ) -> None:
        for name, metric in self.metrics.items():
            self.log(name, metric.compute(), prog_bar=True)

    def test_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int
                  ):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self,
                       outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
                       ):
        self.validation_epoch_end(outputs)

    def _forward(self,
                 session,
                 mask
                 ):
        return self.model(session, mask)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta_1, self.beta_2)
        )
