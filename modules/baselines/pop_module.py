from typing import Dict

import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch.nn.parameter import Parameter

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from metrics.container.metrics_container import MetricsContainer
from modules.metrics_trait import MetricsTrait
from pytorch_lightning import core as pl

from modules.util.module_util import get_padding_mask, build_eval_step_return_dict
from modules.util.noop_optimizer import NoopOptimizer
from tokenization.tokenizer import Tokenizer


class PopModule(MetricsTrait, pl.LightningModule):

    def __init__(self,
                 item_vocab_size: int,
                 tokenizer: Tokenizer,
                 metrics: MetricsContainer):

        super(PopModule, self).__init__()
        self.item_vocab_size = item_vocab_size
        self.tokenizer = tokenizer
        self.metrics = metrics

        self.popularities = Parameter(torch.zeros(self.item_vocab_size, device=self.device), requires_grad=False)

    def on_train_start(self) -> None:
        # TODO: This is a really hacky way to end training after a single epoch, can we do better?
        if self.trainer.max_epochs > 1:
            rank_zero_warn(
                f"When training the POP baseline, "
                f"'trainer.max_epochs' should be set to 1 (but was {self.trainer.max_epochs}).")

    def get_metrics(self) -> MetricsContainer:
        return self.metrics

    def forward(self, input_seq: torch.tensor):
        batch_size = input_seq.shape[0]
        predictions = torch.zeros((batch_size, self.item_vocab_size), device=self.device)
        pop = torch.argmax(self.popularities)
        predictions[:, pop] = 1
        return predictions

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]

        if len(targets.shape) == 1:
            targets = targets.unsqueeze(dim=1)

        mask = get_padding_mask(input_seq, self.tokenizer)
        full_sequence = torch.cat([input_seq * mask, targets], dim=1)
        self.popularities += torch.bincount(full_sequence[full_sequence > 0].flatten(), minlength=self.item_vocab_size)

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
