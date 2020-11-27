from typing import Union, Dict, List, Optional

import pytorch_lightning as pl
import torch

from data.datasets import ITEM_SEQ_ENTRY_NAME, USER_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, \
    NEGATIVE_SAMPLES_ENTRY_NAME, TARGET_ENTRY_NAME
from models.caser.caser_model import CaserModel
from modules.util.module_util import build_eval_step_return_dict
from tokenization.tokenizer import Tokenizer


class CaserModule(pl.LightningModule):

    @staticmethod
    def get_users_from_batch(batch: Dict[str, torch.Tensor]
                             ) -> Optional[torch.Tensor]:
        return batch[USER_ENTRY_NAME] if USER_ENTRY_NAME in batch else None

    def __init__(self,
                 model: CaserModel,
                 tokenizer: Tokenizer,
                 learning_rate: float,
                 weight_decay: float,
                 ):
        super().__init__()

        self.model = model

        self.tokenizer = tokenizer

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        users = CaserModule.get_users_from_batch(batch)
        pos_items = batch[POSITIVE_SAMPLES_ENTRY_NAME]
        neg_items = batch[NEGATIVE_SAMPLES_ENTRY_NAME]

        pos_logits, neg_logits = self.model(input_seq, users, pos_items, neg_items)

        loss = self._calc_loss(pos_logits, neg_logits)
        return {
            'loss': loss
        }

    def _calc_loss(self,
                   pos_logits: torch.Tensor,
                   neg_logits: torch.Tensor
                   ) -> torch.Tensor:
        # TODO: check: this could be the sas loss
        positive_loss = - torch.mean(torch.log(torch.sigmoid(pos_logits)))
        negative_loss = - torch.mean(torch.log(1 - torch.sigmoid(neg_logits)))
        return positive_loss + negative_loss

    # FIXME: a lot of copy paste code
    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        users = CaserModule.get_users_from_batch(batch)
        targets = batch[TARGET_ENTRY_NAME]

        batch_size = input_seq.size()[0]

        # provide items that the target item will be ranked against
        # TODO (AD) refactor this into a composable class to allow different strategies for item selection
        device = input_seq.device
        items_to_rank = torch.as_tensor(self.tokenizer.get_vocabulary().ids(), dtype=torch.long, device=device)
        items_to_rank = items_to_rank.repeat([batch_size, 1])

        prediction = self.model(input_seq, users, items_to_rank)

        return build_eval_step_return_dict(prediction, targets)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )