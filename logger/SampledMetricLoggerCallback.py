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
        # never sample targets
        weight[:, targets] = 0.
        # ... or items in the sequence
        weight[:, input_seq] = 0.

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
            pl_module.log(name, metric.compute(), prog_bar=True)
