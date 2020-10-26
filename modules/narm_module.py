from typing import List, Union, Dict

import torch

import pytorch_lightning as pl

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from metrics.utils.metric_utils import build_metrics
from models.narm.narm_model import NarmModel
from modules.bert4rec_module import get_padding_mask
from tokenization.tokenizer import Tokenizer


class NarmModule(pl.LightningModule):

    def __init__(self,
                 model: NarmModel,
                 batch_size: int,
                 learning_rate: float,
                 beta_1: float,
                 beta_2: float,
                 tokenizer: Tokenizer,
                 batch_first: bool,
                 metrics: torch.nn.ModuleDict
                 ):
        """
        Initializes the Narm Module.
        """
        super().__init__()
        self.model = model

        self.batch_size = batch_size
        self.batch_first = batch_first
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tokenizer = tokenizer

        self.metrics = metrics

    def training_step(self, batch, batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]

        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        logits = self.model(input_seq, padding_mask, batch_idx)
        loss = torch.nn.functional.cross_entropy(logits, target)

        return {
            "loss": loss
        }

    def validation_step(self, batch, batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]

        # calc the padding mask
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        logits = self.model(input_seq, padding_mask, batch_idx)

        for name, metric in self.metrics.items():
            step_value = metric(logits, target)
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
