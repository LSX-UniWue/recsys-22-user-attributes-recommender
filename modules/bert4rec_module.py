from abc import abstractmethod

import torch
import pytorch_lightning as pl

from typing import Union, Dict, Optional, Any

from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, POSITION_IDS
from metrics.container.metrics_container import MetricsContainer
from modules import LOG_KEY_VALIDATION_LOSS, LOG_KEY_TEST_LOSS, LOG_KEY_TRAINING_LOSS
from modules.metrics_trait import MetricsTrait
from modules.util.module_util import get_padding_mask, convert_target_to_multi_hot, build_eval_step_return_dict
from tokenization.tokenizer import Tokenizer
from models.bert4rec.bert4rec_model import BERT4RecModel, BERT4RecBaseModel
from utils.hyperparameter_utils import save_hyperparameters


class BERT4RecBaseModule(MetricsTrait, pl.LightningModule):
    """
    BERT4Rec module for the BERT4Rec model
    """

    @staticmethod
    def get_position_ids(batch: Dict[str, torch.Tensor]
                         ) -> Optional[torch.Tensor]:
        return batch[POSITION_IDS] if POSITION_IDS in batch else None

    @save_hyperparameters
    def __init__(self,
                 model: BERT4RecBaseModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.99,
                 beta_2: float = 0.998,
                 weight_decay: float = 0.001,
                 num_warmup_steps: int = 10000
                 ):
        super().__init__()
        self.model = model

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps

        self.item_tokenizer = item_tokenizer
        self.metrics = metrics
        self.save_hyperparameters(self.hyperparameters)

    def get_metrics(self) -> MetricsContainer:
        return self.metrics

    def forward(self,
                batch: Dict[str, torch.Tensor],
                batch_idx: int
                ) -> torch.Tensor:
        return self._forward_internal(batch, batch_idx)

    @abstractmethod
    def _forward_internal(self, batch: Dict[str, torch.Tensor],
                          batch_idx: int
                          ) -> torch.Tensor:
        pass

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]
        position_ids = BERT4RecModule.get_position_ids(batch)

        # calc the padding mask
        padding_mask = get_padding_mask(sequence=input_seq, tokenizer=self.item_tokenizer)

        # call the model
        prediction_logits = self.model(input_seq, padding_mask=padding_mask, position_ids=position_ids)

        masked_lm_loss = self._calc_loss(prediction_logits, target)
        self.log(LOG_KEY_TRAINING_LOSS, masked_lm_loss, prog_bar=False)
        return {
            'loss': masked_lm_loss
        }

    def _get_prediction_for_masked_item(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target_mask = input_seq.eq(self.item_tokenizer.mask_token_id)

        # get predictions for all seq steps
        prediction = self(batch, batch_idx)
        # extract the relevant seq steps, where the mask was set, here only one mask per sequence steps exists
        return prediction[target_mask]

    def _calc_loss(self,
                   prediction_logits: torch.Tensor,
                   target: torch.Tensor,
                   is_eval: bool = False
                   ) -> torch.Tensor:
        target_size = len(target.size())
        vocab_size = prediction_logits.size()[-1]
        is_basket_recommendation = target_size > 1 if is_eval else target_size > 2
        if is_basket_recommendation:
            target = convert_target_to_multi_hot(target, vocab_size, self.item_tokenizer.pad_token_id)
            loss_fnc = nn.BCEWithLogitsLoss()
            return loss_fnc(prediction_logits, target)

        # handle single item per sequence step
        loss_func = nn.CrossEntropyLoss(ignore_index=self.item_tokenizer.pad_token_id)

        flatten_predictions = prediction_logits.view(-1, vocab_size)
        flatten_targets = target.view(-1)
        return loss_func(flatten_predictions, flatten_targets)

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
            * `POSITION_IDS` a tensor of size (N, S) containing the position ids for the provided sequence

        A padding mask will be generated on the fly, and also the masking of items

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.
        """
        return self._eval_step(batch, batch_idx)

    def _eval_step(self,
                   batch: Dict[str, torch.Tensor],
                   batch_idx: int,
                   is_test: bool = False
                   ) -> Dict[str, torch.Tensor]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]

        # get predictions for all seq steps
        prediction = self._get_prediction_for_masked_item(batch, batch_idx)

        loss = self._calc_loss(prediction, targets, is_eval=True)
        self.log(LOG_KEY_TEST_LOSS if is_test else LOG_KEY_VALIDATION_LOSS, loss, prog_bar=True)

        # when we have multiple target per sequence step, we have to provide a mask for the paddings applied to
        # the target tensor
        mask = None if len(targets.size()) == 1 else ~ targets.eq(self.item_tokenizer.pad_token_id)

        return build_eval_step_return_dict(input_seq, prediction, targets, mask=mask)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, is_test=True)

    def predict(self,
                batch: Any,
                batch_idx: int,
                dataloader_idx: Optional[int] = None
                ) -> torch.Tensor:
        return self._get_prediction_for_masked_item(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     betas=(self.beta_1, self.beta_2))

        if self.num_warmup_steps > 0:
            num_warmup_steps = self.num_warmup_steps

            def _learning_rate_scheduler(step: int) -> float:
                warmup_percent_done = step / num_warmup_steps
                # the learning rate should be reduce by step/warmup-step if in warmup-steps,
                # else the learning rate is fixed
                return min(1.0, warmup_percent_done)

            scheduler = LambdaLR(optimizer, _learning_rate_scheduler)

            schedulers = [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'strict': True,
                }
            ]
            return [optimizer], schedulers
        return [optimizer]


class BERT4RecModule(BERT4RecBaseModule):
    """
    BERT4Rec module for the BERT4Rec model
    """

    @save_hyperparameters
    def __init__(self,
                 model: BERT4RecModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 learning_rate: float = 0.001,
                 beta_1: float = 0.99,
                 beta_2: float = 0.998,
                 weight_decay: float = 0.001,
                 num_warmup_steps: int = 10000
                 ):
        super().__init__(model=model,
                         item_tokenizer=item_tokenizer,
                         metrics=metrics,
                         learning_rate=learning_rate,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         weight_decay=weight_decay,
                         num_warmup_steps=num_warmup_steps)

    def _forward_internal(self, batch: Dict[str, torch.Tensor],
                          batch_idx: int
                          ) -> torch.Tensor:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        position_ids = BERT4RecModule.get_position_ids(batch)

        # calc the padding mask
        padding_mask = get_padding_mask(sequence=input_seq, tokenizer=self.item_tokenizer)

        # call the model
        return self.model(input_seq, padding_mask=padding_mask, position_ids=position_ids)
