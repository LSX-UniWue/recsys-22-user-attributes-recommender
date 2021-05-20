from typing import Union, Dict, Optional

import torch

import pytorch_lightning as pl
from asme.losses.losses import SequenceRecommenderLoss, CrossEntropyLoss
from pytorch_lightning.core.decorators import auto_move_data

from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from asme.metrics.container.metrics_container import MetricsContainer
from asme.modules import LOG_KEY_VALIDATION_LOSS, LOG_KEY_TRAINING_LOSS
from asme.modules.metrics_trait import MetricsTrait
from asme.modules.util.module_util import get_padding_mask, build_eval_step_return_dict, get_additional_meta_data
from asme.tokenization.tokenizer import Tokenizer
from asme.utils.hyperparameter_utils import save_hyperparameters


class NextItemPredictionTrainingModule(MetricsTrait, pl.LightningModule):
    """
    A training module for models that get a sequence and must predict the next item in the sequence

    Supported models are:
    - NARM
    - RNN
    """

    @save_hyperparameters
    def __init__(self,
                 model: SequenceRecommenderModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.99,
                 beta_2: float = 0.998,
                 weight_decay: float = 0,
                 loss_function: SequenceRecommenderLoss = CrossEntropyLoss()
                 ):
        """
        Initializes the training module.
        """
        super().__init__()
        self.model = model

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay

        self.item_tokenizer = item_tokenizer

        self.metrics = metrics

        self.loss_function = loss_function

        self.save_hyperparameters(self.hyperparameters)

    def get_metrics(self) -> MetricsContainer:
        return self.metrics

    @auto_move_data
    def forward(self,
                batch: Dict[str, torch.Tensor],
                batch_idx: int
                ) -> torch.Tensor:
        """
        Applies the recommender model on a batch of sequences and returns logits for every sample in the batch.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),

        A padding mask will be calculated on the fly, based on the `self.tokenizer` of the module.

        :param batch: a batch.
        :param batch_idx: the batch number.

        :return: a tensor with logits for every batch of size (N, I)

        Where N is the batch size, S the max sequence length, and I the item vocabulary size.
        """
        additional_meta_data = get_additional_meta_data(self.model, batch)

        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        padding_mask = get_padding_mask(input_seq, self.item_tokenizer)

        return self.model(input_seq, padding_mask, **additional_meta_data)

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        """
        Performs a validation step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
            * `TARGET_ENTRY_NAME`: a tensor of size (N) with the target items,

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.

        where N is the batch size and S the max sequence length.
        """

        logits = self(batch, batch_idx)
        target = batch[TARGET_ENTRY_NAME]

        loss = self._calc_loss(logits, target)
        self.log(LOG_KEY_TRAINING_LOSS, loss)
        return {
            "loss": loss
        }

    def _calc_loss(self,
                   logits: torch.Tensor,
                   target_tensor: torch.Tensor
                   ) -> torch.Tensor:

        return self.loss_function(target_tensor, logits)

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        """
        Performs a validation step on a batch of sequences and returns the entries according
        to `build_eval_step_return_dict`.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
            * `TARGET_ENTRY_NAME`: a tensor of size (N) with the target items,

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.

        where N is the batch size and S the max sequence length.
        """

        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]

        logits = self(batch, batch_idx)

        loss = self._calc_loss(logits, target)
        self.log(LOG_KEY_VALIDATION_LOSS, loss, prog_bar=True)

        mask = None if len(target.size()) == 1 else ~ target.eq(self.item_tokenizer.pad_token_id)

        return build_eval_step_return_dict(input_seq, logits, target, mask=mask)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def predict(self,
                batch:  Dict[str, torch.Tensor],
                batch_idx: int,
                dataloader_idx: Optional[int] = None
                ) -> torch.Tensor:
        return self(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                betas=(self.beta_1, self.beta_2),
                                weight_decay=self.weight_decay)
