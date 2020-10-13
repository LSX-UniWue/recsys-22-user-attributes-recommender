from typing import List, Union, Dict

import torch

import pytorch_lightning as pl

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss
from metrics.ranking_metrics import RecallAtMetric, MRRAtMetric
from modules.bert4rec_module import get_padding_mask
from models.sasrec.sas_rec_model import SASRecModel
from tokenization.tokenizer import Tokenizer


class SASRecModule(pl.LightningModule):

    def __init__(self,
                 model: SASRecModel,
                 batch_size: int,
                 learning_rate: float,
                 beta_1: float,
                 beta_2: float,
                 tokenizer: Tokenizer,
                 batch_first: bool,
                 metrics_k: List[int]
                 ):
        """
        inits the SASRec module
        :param training_config: all training configurations
        :param model_config: all model configurations
        """
        super().__init__()
        self.model = model

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tokenizer = tokenizer
        self.batch_first = batch_first

        metrics = {}

        for k in metrics_k:
            metrics[f"recall_at_{k}"] = RecallAtMetric(k)
            metrics[f"mrr_at_{k}"] = MRRAtMetric(k)

        self.metrics = torch.nn.ModuleDict(modules=metrics)

    def training_step(self, batch, batch_idx):
        input_seq = batch['session']
        pos = batch['positive_samples']
        neg = batch['negative_samples']

        if self.batch_first:
            input_seq = input_seq.transpose(1, 0)
            pos = pos.transpose(1, 0)
            neg = neg.transpose(1, 0)

        padding_mask = get_padding_mask(input_seq, self.tokenizer)

        pos_logits, neg_logits = self.model(input_seq, pos, neg_items=neg, padding_mask=padding_mask)

        loss_func = SASRecBinaryCrossEntropyLoss()
        loss = loss_func(pos_logits, neg_logits, mask=padding_mask.transpose(0, 1))
        # the original code
        # (https://github.com/kang205/SASRec/blob/641c378fcfac265ea8d1e5fe51d4d53eb892d1b4/model.py#L92)
        # adds regularization losses, but they are empty, as far as I can see (dzo)
        # TODO: check

        return {
            "loss": loss
        }

    def validation_step(self, batch, batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]

        if self.batch_first:
            input_seq = input_seq.transpose(1, 0)
            batch_size = input_seq.size()[1]
        else:
            batch_size = input_seq.size()[0]

        padding_mask = get_padding_mask(input_seq, self.tokenizer)
        # the first entry in each tensor

        # provide items that the target item will be ranked against
        # TODO (AD) refactor this into a composable class to allow different strategies for item selection
        device = input_seq.device
        items_to_rank = torch.as_tensor(self.tokenizer.get_vocabulary().ids(), dtype=torch.long, device=device)
        items_to_rank = items_to_rank.repeat([batch_size, 1])
        items_to_rank = items_to_rank.transpose(1, 0)

        prediction = self.model(input_seq, items_to_rank, padding_mask=padding_mask)

        prediction = prediction.transpose(1, 0)

        for name, metric in self.metrics.items():
            step_value = metric(prediction, targets)
            self.log(name, step_value, prog_bar=True)

    def validation_epoch_end(self, outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]) -> None:
        for name, metric in self.metrics.items():
            self.log(name, metric.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                betas=(self.beta_1, self.beta_2))
