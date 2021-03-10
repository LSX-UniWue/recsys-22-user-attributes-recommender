import torch

from typing import Union, Dict, Optional, List

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from metrics.container.metrics_container import MetricsContainer
from modules import LOG_KEY_VALIDATION_LOSS, LOG_KEY_TEST_LOSS, LOG_KEY_TRAINING_LOSS
from modules.util.module_util import get_padding_mask, build_eval_step_return_dict
from tokenization.tokenizer import Tokenizer
from models.kebert4rec.kebert4rec_model import KeBERT4RecModel
from modules.bert4rec_module import BERT4RecBaseModule
from utils.hyperparameter_utils import save_hyperparameters


class KeBERT4RecModule(BERT4RecBaseModule):

    @save_hyperparameters
    def __init__(self,
                 model: KeBERT4RecModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 additional_attributes: List[str],
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
        self.attributes = additional_attributes

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]

        position_ids = KeBERT4RecModule.get_position_ids(batch)

        attribute_sequences = self.get_attribute_sequences(batch)

        # calc the padding mask
        padding_mask = get_padding_mask(input_seq,
                                        tokenizer=self.item_tokenizer)

        # call the model
        prediction_logits = self.model(input_seq, padding_mask=padding_mask, position_ids=position_ids, **attribute_sequences)

        masked_lm_loss = self._calc_loss(prediction_logits, target)
        self.log(LOG_KEY_TRAINING_LOSS, masked_lm_loss, prog_bar=False)
        return {
            'loss': masked_lm_loss,

        }

    def get_attribute_sequences(self, batch):
        attribute_sequences = {}
        for attribute in self.attributes:
            attribute_sequences[attribute] = batch[attribute]
        return attribute_sequences

    def _eval_step(self,
                   batch: Dict[str, torch.Tensor],
                   batch_idx: int,
                   is_test: bool = False
                   ) -> Dict[str, torch.Tensor]:

        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]
        position_ids = KeBERT4RecModule.get_position_ids(batch)

        attribute_sequences = self.get_attribute_sequences(batch)

        target_mask = input_seq.eq(self.item_tokenizer.mask_token_id)

        # handle basket training and evaluation
        if len(target_mask.size()) > 2:
            target_mask = target_mask.max(dim=-1)[0]

        # after adding the mask token we can calculate the padding mask
        padding_mask = get_padding_mask(input_seq, self.item_tokenizer)

        # get predictions for all seq steps
        prediction = self.model(input_seq, padding_mask=padding_mask, position_ids=position_ids, **attribute_sequences)
        # extract the relevant seq steps, where the mask was set, here only one mask per sequence steps exists
        prediction = prediction[target_mask]

        loss = self._calc_loss(prediction, targets, is_eval=True)
        self.log(LOG_KEY_TEST_LOSS if is_test else LOG_KEY_VALIDATION_LOSS, loss, prog_bar=True)

        # when we have multiple target per sequence step, we have to provide a mask for the paddings applied to
        # the target tensor
        mask = None if len(targets.size()) == 1 else ~ targets.eq(self.item_tokenizer.pad_token_id)

        return build_eval_step_return_dict(input_seq, prediction, targets, mask=mask)
