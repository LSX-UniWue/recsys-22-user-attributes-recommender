from typing import Union, Dict, Optional

import pytorch_lightning as pl
import torch
from torch import nn

from data.datasets import ITEM_SEQ_ENTRY_NAME, USER_ENTRY_NAME, TARGET_ENTRY_NAME
from metrics.container.metrics_container import MetricsContainer
from models.basket.nnrec.nnrec_model import NNRecModel
from modules.util.module_util import build_eval_step_return_dict, convert_target_to_multi_hot
from tokenization.tokenizer import Tokenizer
from utils.hyperparameter_utils import save_hyperparameters


class NNRecModule(pl.LightningModule):

    """
    module to train a NNRec model
    """

    @staticmethod
    def get_users_from_batch(batch: Dict[str, torch.Tensor]
                             ) -> Optional[torch.Tensor]:
        return batch[USER_ENTRY_NAME] if USER_ENTRY_NAME in batch else None

    @save_hyperparameters
    def __init__(self,
                 model: NNRecModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.99,
                 beta_2: float = 0.998,
                 weight_decay: float = 0.01
                 ):
        super().__init__()

        self.model = model

        self.tokenizer = item_tokenizer

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.metrics = metrics
        self.save_hyperparameters(self.hyperparameters)

    def forward(self,
                batch: Dict[str, torch.Tensor],
                batch_idx: int
                ) -> torch.Tensor:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        user = NNRecModule.get_users_from_batch(batch)
        return self.model(input_seq, user=user)

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        """
        Performs a training step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
            * `TARGET_ENTRY_NAME` a tensor of size (N, B) containing the target items
        Optional entries are:
            * `USER_ENTRY_NAME` a tensor of size (N) containing the user ids for the sequences

        Where N is the batch size, S the max sequence length and B is the maximum basket size.

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: the total loss
        """

        target = batch[TARGET_ENTRY_NAME]
        logits = self(batch, batch_idx)

        loss = self._calc_loss(logits, target)
        return {
            'loss': loss
        }

    def _calc_loss(self,
                   logits: torch.Tensor,
                   target: torch.Tensor
                   ) -> torch.Tensor:
        loss_fnc = nn.BCEWithLogitsLoss()
        target_tensor = convert_target_to_multi_hot(target, len(self.tokenizer), self.tokenizer.pad_token_id)
        return loss_fnc(logits, target_tensor)

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        """
        Performs a validation step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
            * `TARGET_ENTRY_NAME`: a tensor of size (N) with the target items,
        Optional entries are:
            * `USER_ENTRY_NAME` a tensor of size (N) containing the user id for the provided sequences

        A padding mask will be generated on the fly, and also the masking of items

        Where N is the batch size and S the max sequence length.

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.
        """
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]

        prediction = self(batch, batch_idx)

        mask = ~ target.eq(self.tokenizer.pad_token_id)
        return build_eval_step_return_dict(input_seq, prediction, target, mask=mask)

    def test_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int
                  ) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.beta_1, self.beta_2)
        )
