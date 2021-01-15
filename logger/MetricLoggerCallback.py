
from pytorch_lightning import Callback
from torch.nn import ModuleDict

from modules.constants import RETURN_KEY_PREDICTIONS, RETURN_KEY_TARGETS, RETURN_KEY_MASK


class MetricLoggerCallback(Callback):

    def __init__(self, metrics: ModuleDict):
        self.metrics = metrics

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        targets = outputs[RETURN_KEY_TARGETS]
        predictions = outputs[RETURN_KEY_PREDICTIONS]
        mask = outputs[RETURN_KEY_MASK] if RETURN_KEY_MASK in outputs else None
        for name, metric in self.metrics.items():
            step_value = metric(predictions, targets, mask=mask)
            pl_module.log(name, step_value, prog_bar=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        for name, metric in self.metrics.items():
            pl_module.log(name, metric.compute(), prog_bar=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        return self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self, trainer, pl_module):
        return self.on_validation_epoch_end(trainer, pl_module)