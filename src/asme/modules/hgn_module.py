from typing import Union, Dict, Optional

import pytorch_lightning as pl
import torch

from asme.modules import LOG_KEY_TRAINING_LOSS
from data.datasets import ITEM_SEQ_ENTRY_NAME, USER_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, \
    NEGATIVE_SAMPLES_ENTRY_NAME, TARGET_ENTRY_NAME
from asme.metrics.container.metrics_container import MetricsContainer
from asme.models.hgn.hgn_model import HGNModel
from asme.modules.metrics_trait import MetricsTrait
from asme.modules.util.module_util import build_eval_step_return_dict, get_additional_meta_data
from asme.tokenization.tokenizer import Tokenizer
from asme.utils.hyperparameter_utils import save_hyperparameters


# FIXME: merge with SequenceNextItemPredictionTrainingModule
class HGNModule(MetricsTrait, pl.LightningModule):

    @staticmethod
    def get_users_from_batch(batch: Dict[str, torch.Tensor]
                             ) -> Optional[torch.Tensor]:
        return batch[USER_ENTRY_NAME] if USER_ENTRY_NAME in batch else None

    @save_hyperparameters
    def __init__(self,
                 model: HGNModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-3
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

        pos_items = batch[POSITIVE_SAMPLES_ENTRY_NAME]
        neg_items = batch[NEGATIVE_SAMPLES_ENTRY_NAME]
        target = torch.cat((pos_items, neg_items), 1)

        additional_metadata = get_additional_meta_data(self.model, batch)
        logits = self.model(input_seq, target, padding_mask=None, for_pred=False, **additional_metadata)

        (targets_prediction, negatives_prediction) = torch.split(
            logits, [pos_items.size(1), neg_items.size(1)], dim=1)

        # Compute the BPR loss
        loss = - torch.log(torch.sigmoid(targets_prediction - negatives_prediction) + 1e-8)
        loss = torch.mean(torch.sum(loss))

        self.log(LOG_KEY_TRAINING_LOSS, loss)

        return {
            'loss': loss
        }


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

        Where N is the batch size and S the max sequence length.

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.
        """
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        additional_metadata = get_additional_meta_data(self.model, batch)
        targets = batch[TARGET_ENTRY_NAME]

        batch_size = input_seq.size()[0]

        # provide items that the target item will be ranked against
        # TODO (AD) refactor this into a composable class to allow different strategies for item selection
        device = input_seq.device

        items_to_rank = torch.as_tensor(self.item_tokenizer.get_vocabulary().ids(), dtype=torch.long, device=device)
        #items_to_rank = items_to_rank.repeat([batch_size, 1]) # (N, I)
        # TODO items_to_rank muss eindimensional sein! shape(I)

        prediction = self.model(input_seq, items_to_rank, padding_mask=None, for_pred=True, **additional_metadata) # (N, I) # True

       #(targets_prediction, negatives_prediction) = torch.split(
      #      logits, [pos_items.size(1), neg_items.size(1)], dim=1)

        # Compute the BPR loss
       # loss = -torch.log(torch.sigmoid(targets_prediction - negatives_prediction) + 1e-8)
     #   loss = torch.mean(torch.sum(loss))

        return build_eval_step_return_dict(input_seq, prediction, targets)


    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
