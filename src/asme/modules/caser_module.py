from typing import Union, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch

from data.datasets import ITEM_SEQ_ENTRY_NAME, USER_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, \
    NEGATIVE_SAMPLES_ENTRY_NAME, TARGET_ENTRY_NAME
from asme.metrics.container.metrics_container import MetricsContainer
from asme.models.caser.caser_model import CaserModel
from asme.modules.metrics_trait import MetricsTrait
from asme.modules.util.module_util import build_eval_step_return_dict
from asme.tokenization.tokenizer import Tokenizer
from asme.utils.hyperparameter_utils import save_hyperparameters


class CaserModule(MetricsTrait, pl.LightningModule):
    """

    The module for caser, that defines the training for the Caser model

    """

    @staticmethod
    def get_users_from_batch(batch: Dict[str, torch.Tensor]
                             ) -> Optional[torch.Tensor]:
        return batch[USER_ENTRY_NAME] if USER_ENTRY_NAME in batch else None

    @save_hyperparameters
    def __init__(self,
                 model: CaserModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.001
                 ):
        super().__init__()

        self.model = model
        self.item_tokenizer = item_tokenizer

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.metrics = metrics

        self.save_hyperparameters(self.hyperparameters)

    def get_metrics(self) -> MetricsContainer:
        return self.metrics

    def forward(self,
                sequence: torch.Tensor,
                user: torch.Tensor,
                pos_items: torch.Tensor,
                neg_items: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        return self.model(sequence, user, pos_items, neg_items)

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        """
        Performs a training step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
            * `POSITIVE_SAMPLES_ENTRY_NAME`: a tensor of size (N) containing the next sequence items (pos examples)
            * `NEGATIVE_SAMPLES_ENTRY_NAME`: a tensor of size (N) containing a negative item (sampled)
        Optional entries are:
            * `USER_ENTRY_NAME` a tensor of size (N) containing the user id for the provided sequences

        Where N is the batch size and S the max sequence length.

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: the total loss
        """
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
        targets = batch[TARGET_ENTRY_NAME]
        prediction = self.predict(batch, batch_idx)

        return build_eval_step_return_dict(input_seq, prediction, targets)

    def predict(self,
                batch: Dict[str, torch.Tensor],
                batch_idx: int,
                dataloader_idx: Optional[int] = None
                ) -> torch.Tensor:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        users = CaserModule.get_users_from_batch(batch)

        batch_size = input_seq.size()[0]

        # provide items that the target item will be ranked against
        # TODO (AD) refactor this into a composable class to allow different strategies for item selection
        device = input_seq.device
        items_to_rank = torch.as_tensor(self.item_tokenizer.get_vocabulary().ids(), dtype=torch.long, device=device)
        items_to_rank = items_to_rank.repeat([batch_size, 1])

        return self(input_seq, users, items_to_rank)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )