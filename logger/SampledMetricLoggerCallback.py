from typing import List

import torch
from pytorch_lightning import Callback
from torch.nn import ModuleDict

from modules.constants import RETURN_KEY_PREDICTIONS, RETURN_KEY_TARGETS, RETURN_KEY_MASK, RETURN_KEY_SEQUENCE


class SampledMetricLoggerCallback(Callback):

    def __init__(self,
                 metrics: ModuleDict,
                 item_probabilities: List[float],
                 num_negative_samples: int
                 ):
        self.metrics = metrics
        self.item_probabilities = item_probabilities
        self.num_negative_samples = num_negative_samples

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        targets = outputs[RETURN_KEY_TARGETS]
        input_seq = outputs[RETURN_KEY_SEQUENCE]
        predictions = outputs[RETURN_KEY_PREDICTIONS]
        mask = outputs[RETURN_KEY_MASK] if RETURN_KEY_MASK in outputs else None

        weight = torch.Tensor(self.item_probabilities)
        weight = weight.unsqueeze(0).repeat(input_seq.size()[0], 1)

        # FIXME (BUG): this does not work as expected. It sets the all targets in a batch across all weights to 0..
        #  Same for input_seq. In consequence more and more items are set to 0 with increasing batch size :-D

        # never sample targets
        weight[:, targets] = 0.

        # we want to use scatter to set the items contained in the input to 0 for every row in the batch
        src = torch.ones_like(input_seq).to(torch.long)
        mask = torch.zeros_like(weight).to(torch.long)

        # calculate a mask where 1. signals that the item should get 0. probability since it occurs in the input
        # sequence.
        mask.scatter_(1, input_seq, src)
        weight[mask.to(dtype=torch.bool)] = 0.

        sampled_negatives = torch.multinomial(weight, num_samples=self.num_negative_samples)
        target_batched = targets.unsqueeze(1)
        sampled_items = torch.cat([target_batched, sampled_negatives], dim=1)

        positive_item_mask = sampled_items.eq(target_batched).to(dtype=predictions.dtype)
        # FIXME: fix positive_item_mask with mask
        sampled_predictions = predictions.gather(1, sampled_items)

        for name, metric in self.metrics.items():
            step_value = metric(sampled_predictions, positive_item_mask)
            pl_module.log(f"{name}_(sampled)", step_value, prog_bar=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        for name, metric in self.metrics.items():
            pl_module.log(f"{name}_(sampled,{self.num_negative_samples})", metric.compute(), prog_bar=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        return self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self, trainer, pl_module):
        return self.on_validation_epoch_end(trainer, pl_module)
