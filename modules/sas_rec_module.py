from typing import Union, Dict, Optional

import torch

import pytorch_lightning as pl

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME
from losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss
from metrics.container.metrics_container import MetricsContainer
from modules.metrics_trait import MetricsTrait
from modules.util.module_util import get_padding_mask, build_eval_step_return_dict
from models.sasrec.sas_rec_model import SASRecModel
from tokenization.tokenizer import Tokenizer
from utils.hyperparameter_utils import save_hyperparameters


class SASRecModule(MetricsTrait, pl.LightningModule):

    """
    the module for training a SASRec model
    """

    @save_hyperparameters
    def __init__(self,
                 model: SASRecModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.99,
                 beta_2: float = 0.998
                 ):
        """
        inits the SASRec module
        :param model: the model to train
        :param learning_rate: the learning rate
        :param beta_1: the beta1 of the adam optimizer
        :param beta_2: the beta2 of the adam optimizer
        :param item_tokenizer: the tokenizer
        :param metrics: metrics to compute on validation/test
        """
        super().__init__()
        self.model = model

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.item_tokenizer = item_tokenizer
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
            * `POSITIVE_SAMPLES_ENTRY_NAME`: a tensor of size (N) containing the next sequence items (pos examples),
            * `NEGATIVE_SAMPLES_ENTRY_NAME`: a tensor of size (N) containing a negative item (sampled)

        Where N is the batch size and S the max sequence length.

        A padding mask will be generated on the fly, and also the masking of items

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: the total loss
        """

        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        pos = batch[POSITIVE_SAMPLES_ENTRY_NAME]
        neg = batch[NEGATIVE_SAMPLES_ENTRY_NAME]

        padding_mask = get_padding_mask(input_seq, self.item_tokenizer)

        pos_logits, neg_logits = self.model(input_seq, pos, neg_items=neg, padding_mask=padding_mask)

        loss_func = SASRecBinaryCrossEntropyLoss()
        loss = loss_func(pos_logits, neg_logits, mask=padding_mask)
        # TODO: check: the original code
        # (https://github.com/kang205/SASRec/blob/641c378fcfac265ea8d1e5fe51d4d53eb892d1b4/model.py#L92)
        # adds regularization losses, but they are empty, as far as I can see (dzo)

        return {
            "loss": loss
        }

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        """
        Performs a validation step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
            * `TARGET_ENTRY_NAME`: a tensor of size (N) containing the next item of the provided sequence

        Where N is the batch size and S the max sequence length.

        A padding mask will be generated on the fly, and also the masking of items

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.
        """
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]

        prediction = self.predict(batch, batch_idx)

        return build_eval_step_return_dict(input_seq, prediction, targets)

    def test_step(self,
                  batch: Dict[str, torch.Tensor],
                  batch_idx: int
                  ) -> Dict[str, torch.Tensor]:
        return self.validation_step(batch, batch_idx)

    def predict(self,
                batch: Dict[str, torch.Tensor],
                batch_idx: int,
                dataloader_idx: Optional[int] = None
                ) -> torch.Tensor:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        # calc the padding mask
        padding_mask = get_padding_mask(input_seq, self.item_tokenizer)

        # provide items that the target item will be ranked against
        # TODO (AD) refactor this into a composable class to allow different strategies for item selection
        batch_size = input_seq.size()[0]
        device = input_seq.device
        items_to_rank = torch.as_tensor(self.item_tokenizer.get_vocabulary().ids(), dtype=torch.long, device=device)
        items_to_rank = items_to_rank.repeat([batch_size, 1])

        return self.model(input_seq, items_to_rank, padding_mask=padding_mask)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                betas=(self.beta_1, self.beta_2))
