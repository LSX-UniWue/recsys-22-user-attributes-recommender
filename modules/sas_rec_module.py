from typing import List, Union, Dict, Optional

import torch

import pytorch_lightning as pl

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME
from losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss
from modules.util.module_util import get_padding_mask, build_eval_step_return_dict
from models.sasrec.sas_rec_model import SASRecModel
from tokenization.tokenizer import Tokenizer


class SASRecModule(pl.LightningModule):

    def __init__(self,
                 model: SASRecModel,
                 learning_rate: float,
                 beta_1: float,
                 beta_2: float,
                 tokenizer: Tokenizer,
                 batch_first: bool,
                 metrics: torch.nn.ModuleDict
                 ):
        """
        inits the SASRec module
        :param model: the model to learn
        :param learning_rate: the learning rate
        :param beta_1: the beta1 of the adam optimizer
        :param beta_2: the beta2 of the adam optimizer
        :param tokenizer: the tokenizer
        :param batch_first: True iff the dataloader returns batch_first tensors
        :param metrics: the metrics to use for evaluation
        """
        super().__init__()
        self.model = model

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tokenizer = tokenizer
        self.batch_first = batch_first

        self.metrics = metrics

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        pos = batch[POSITIVE_SAMPLES_ENTRY_NAME]
        neg = batch[NEGATIVE_SAMPLES_ENTRY_NAME]

        if self.batch_first:
            input_seq = input_seq.transpose(1, 0)
            pos = pos.transpose(1, 0)
            neg = neg.transpose(1, 0)

        padding_mask = get_padding_mask(input_seq, self.tokenizer)

        pos_logits, neg_logits = self.model(input_seq, pos, neg_items=neg, padding_mask=padding_mask)

        loss_func = SASRecBinaryCrossEntropyLoss()
        loss = loss_func(pos_logits, neg_logits, mask=padding_mask.transpose(0, 1))
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
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]

        if self.batch_first:
            input_seq = input_seq.transpose(1, 0)
            batch_size = input_seq.size()[1]
        else:
            batch_size = input_seq.size()[0]

        # calc the padding mask
        padding_mask = get_padding_mask(input_seq, self.tokenizer)

        # provide items that the target item will be ranked against
        # TODO (AD) refactor this into a composable class to allow different strategies for item selection
        device = input_seq.device
        items_to_rank = torch.as_tensor(self.tokenizer.get_vocabulary().ids(), dtype=torch.long, device=device)
        items_to_rank = items_to_rank.repeat([batch_size, 1])
        items_to_rank = items_to_rank.transpose(1, 0)

        prediction = self.model(input_seq, items_to_rank, padding_mask=padding_mask)
        prediction = prediction.transpose(1, 0)

        return build_eval_step_return_dict(prediction, targets)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                betas=(self.beta_1, self.beta_2))
