from typing import Union, List, Dict, Optional

import torch

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.core.decorators import auto_move_data

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from models.rnn.rnn_model import RNNSeqItemRecommenderModel
from modules import LOG_KEY_VALIDATION_LOSS
from modules.util.module_util import get_padding_mask, convert_target_to_multi_hot, build_eval_step_return_dict
from tokenization.tokenizer import Tokenizer


class GRUModule(pl.LightningModule):

    def __init__(self,
                 model: RNNSeqItemRecommenderModel,
                 lr: float,
                 beta_1: float,
                 beta_2: float,
                 tokenizer: Tokenizer,
                 ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tokenizer = tokenizer

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        logits = self.forward(batch, batch_idx)

        target = batch[TARGET_ENTRY_NAME]
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
        """
        Performs a validation step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `data.datasets.ITEM_SEQ_ENTRY_NAME`: a tensor of size [BS x S] with the input sequences.
            * `data.datasets.TARGET_ENTRY_NAME`: a tensor of size [BS] with the target items.

        A padding mask will be calculated on the fly, based on the `self.tokenizer` of the module.

        :param batch: a batch.
        :param batch_idx: the batch number.

        :return: A dictionary with entries according to `build_eval_step_return_dict`.
        """

        input_seq = batch[ITEM_SEQ_ENTRY_NAME]

        logits = self(batch, batch_idx)
        target = batch[TARGET_ENTRY_NAME]

        loss = self._calc_loss(logits, target)
        self.log(LOG_KEY_VALIDATION_LOSS, loss, prog_bar=True)

        mask = None if len(target.size()) == 1 else ~ target.eq(self.tokenizer.pad_token_id)

        return build_eval_step_return_dict(input_seq, logits, target, mask=mask)

    def test_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int
                  ):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self,
                       outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
                       ):
        self.validation_epoch_end(outputs)

    @auto_move_data
    def forward(self,
                batch: Dict[str, torch.Tensor],
                batch_idx: int
                ) -> torch.Tensor:
        """
        Applies the RNN model on a batch of sequences and returns logits for every sample in the batch.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size [BS x S]

        A padding mask will be calculated on the fly, based on the `self.tokenizer` of the module.

        :param batch: a batch.
        :param batch_idx: the batch number.

        :return: a tensor with logits for every batch [BS x |I|]
        """
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        return self.model(input_seq, padding_mask)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta_1, self.beta_2)
        )
