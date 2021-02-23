from typing import Dict

import numpy as np
import pytorch_lightning.core as pl
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch.nn.parameter import Parameter

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from metrics.container.metrics_container import MetricsContainer
from modules.metrics_trait import MetricsTrait
from modules.util.module_util import build_eval_step_return_dict, get_padding_mask
from modules.util.noop_optimizer import NoopOptimizer
from tokenization.tokenizer import Tokenizer


def last_item_in_sequence(input_sequence: torch.tensor) -> torch.Tensor:
    """
    Finds the last item in the provided item sequence, i.e the last non-zero element.
    :param input_sequence: A Tensor of shape (B,L) where L is the maximum sequence length
    :return: Tensor of shape (B) containing he last item per sequence (i.e the last non-zero element).
    """
    length = input_sequence.shape[1]
    binary_seq = torch.where(input_sequence > 0, torch.ones_like(input_sequence), torch.zeros_like(input_sequence))
    rising_seq = torch.arange(1, length + 1)
    indices = torch.argmax(binary_seq * rising_seq, dim=1, keepdim=True)
    # Gather the last items per sample
    items = torch.gather(input_sequence, dim=1, index=indices).squeeze()
    return items


class MarkovModule(MetricsTrait, pl.LightningModule):

    def __init__(self,
                 tokenizer: Tokenizer,
                 item_vocab_size: int,
                 metrics: MetricsContainer):

        super(MarkovModule, self).__init__()
        self.tokenizer = tokenizer
        self.item_vocab_size = item_vocab_size
        self.metrics = metrics

        # Transition matrix of dimension item_vocab_size x item_vocab_size. The entry at index (i,j) denotes the
        # probability of item j being bought right after item i has been bought
        # We artificially promote this Tensor to a parameter to make PL save it in the model checkpoints
        self.transition_matrix = Parameter(torch.full((self.item_vocab_size, self.item_vocab_size),
                                                      1 / self.item_vocab_size,
                                                      device=self.device), requires_grad=False)

    def on_train_start(self) -> None:
        if self.trainer.max_epochs > 1:
            rank_zero_warn(
                f"When training the Markov baseline, "
                f"'trainer.max_epochs' should be set to 1 (but is {self.trainer.max_epochs}).")

        self.transition_matrix.fill_(0)

    def on_train_end(self) -> None:
        # If a all entries of a row are zero (i.e. no training data was seen for this item)
        # we use a uniform distribution. This is slow but only executed once, so it should be fine.
        for i in range(self.item_vocab_size):
            row = self.transition_matrix[i]
            if row.sum() == 0:
                row.fill_(1 / self.item_vocab_size)

        # Normalize the sum of each row to 1 so we can interpret it as a probability distribution
        row_sums = self.transition_matrix.sum(dim=1)
        self.transition_matrix /= row_sums.unsqueeze(dim=1)

    def get_metrics(self) -> MetricsContainer:
        return self.metrics

    def forward(self, input_seq: torch.tensor):
        batch_size = input_seq.shape[0]
        last_items = last_item_in_sequence(input_seq)
        # Sadly PyTorch does not have a "choice" implementation, so we cant do this in a fast way
        transition_probabilities = self.transition_matrix[last_items].numpy()
        # We simply predict 0's for all but the most frequently seen item
        predictions = torch.zeros((batch_size, self.item_vocab_size), device=self.device)
        for i in range(batch_size):
            index = np.random.choice(self.item_vocab_size, size=1, p=transition_probabilities[i])
            predictions[i, index] = 1

        return predictions

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]
        seq_length = input_seq.shape[1]

        mask = get_padding_mask(input_seq, self.tokenizer)
        masked = input_seq * mask
        # Gather the last items bought
        items = last_item_in_sequence(masked)
        # Update the transition matrix
        # TODO: This should be possible using scatter instead of treating the transition matrix as a 1d array
        flat_indices = self.item_vocab_size * items + target
        flattened_transition_matrix = self.transition_matrix.view(-1)
        flattened_transition_matrix[flat_indices] += 1

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
