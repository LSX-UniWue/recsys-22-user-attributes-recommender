from typing import List, Union, Dict, Optional

import torch

import pytorch_lightning as pl

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from models.narm.narm_model import NarmModel
from modules import LOG_KEY_VALIDATION_LOSS
from modules.util.module_util import get_padding_mask, convert_target_to_multi_hot, build_eval_step_return_dict
from tokenization.tokenizer import Tokenizer
from torch import nn


class NarmModule(pl.LightningModule):

    def __init__(self,
                 model: NarmModel,
                 batch_size: int,
                 learning_rate: float,
                 beta_1: float,
                 beta_2: float,
                 tokenizer: Tokenizer,
                 batch_first: bool,
                 metrics: torch.nn.ModuleDict
                 ):
        """
        Initializes the Narm Module.
        """
        super().__init__()
        self.model = model

        self.batch_size = batch_size
        self.batch_first = batch_first
        self.learning_rate = learning_rate
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

        logits = self.model(input_seq, padding_mask)
        loss = self._calc_loss(logits, target)

        return {
            "loss": loss
        }

    def _calc_loss(self,
                   logits: torch.Tensor,
                   target_tensor: torch.Tensor
                   ) -> torch.Tensor:
        if len(target_tensor.size()) == 1:
            # only one item per sequence step
            loss_fnc = nn.CrossEntropyLoss()
            return loss_fnc(logits, target_tensor)

        loss_fnc = nn.BCEWithLogitsLoss()
        target_tensor = convert_target_to_multi_hot(target_tensor, len(self.tokenizer), self.tokenizer.pad_token_id)
        return loss_fnc(logits, target_tensor)

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]

        # calc the padding mask
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        logits = self.model(input_seq, padding_mask)

        loss = self._calc_loss(logits, target)
        self.log(LOG_KEY_VALIDATION_LOSS, loss, prog_bar=True)

        mask = None if len(target.size()) == 1 else ~ target.eq(self.tokenizer.pad_token_id)

        return build_eval_step_return_dict(logits, target, mask=mask)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                betas=(self.beta_1, self.beta_2))
