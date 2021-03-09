from typing import Dict

import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch.nn.parameter import Parameter

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from metrics.container.metrics_container import MetricsContainer
from modules.metrics_trait import MetricsTrait
from pytorch_lightning import core as pl

from modules.util.module_util import  build_eval_step_return_dict
from modules.util.noop_optimizer import NoopOptimizer
from tokenization.tokenizer import Tokenizer


class PopModule(MetricsTrait, pl.LightningModule):

    def __init__(self,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer):

        super(PopModule, self).__init__()
        self.item_tokenizer = item_tokenizer
        self.item_vocab_size = len(item_tokenizer)
        self.metrics = metrics

        # We artificially promote this Tensor to a parameter to make PL save it in the model checkpoints
        self.item_frequencies = Parameter(torch.zeros(self.item_vocab_size, device=self.device), requires_grad=False)

    def on_train_start(self) -> None:
        if self.trainer.max_epochs > 1:
            rank_zero_warn(
                f"When training the POP baseline, "
                f"'trainer.max_epochs' should be set to 1 (but is {self.trainer.max_epochs}).")

    def get_metrics(self) -> MetricsContainer:
        return self.metrics

    def forward(self, input_seq: torch.tensor):
        batch_size = input_seq.shape[0]
        # We simply predict 0's for all but the most frequently seen item
        predictions = torch.zeros((batch_size, self.item_vocab_size), device=self.device)
        pop = torch.argmax(self.item_frequencies)
        predictions[:, pop] = 1
        return predictions

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx):
        input_seq = torch.tensor(batch[ITEM_SEQ_ENTRY_NAME])
        self.item_frequencies += torch.bincount(input_seq, minlength=self.item_vocab_size)

        return {'loss': torch.tensor(0., device=self.device)}

    def eval_step(self, batch: Dict[str, torch.tensor], batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]
        prediction = self.forward(input_seq)

        return build_eval_step_return_dict(input_seq, prediction, targets)

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx):
        return self.eval_step(batch, batch_idx)

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx):
        return self.eval_step(batch, batch_idx)

    # Do nothing on backward since we only count occurrences
    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, optimizer_idx: int, *args,
                 **kwargs) -> None:
        pass

    def configure_optimizers(self):
        return NoopOptimizer()
