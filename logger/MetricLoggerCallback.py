from pytorch_lightning import Callback
from pytorch_lightning.core import LightningModule

from data.datasets import TARGET_ENTRY_NAME


def check_has_metrics(module: LightningModule):
    return hasattr(module, 'metrics')

class MetricLoggerCallback(Callback):

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if check_has_metrics(pl_module):
            targets = batch[TARGET_ENTRY_NAME]
            for name, metric in pl_module.metrics.items():
                step_value = metric(outputs, targets)
                pl_module.log(name, step_value, prog_bar=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if check_has_metrics(pl_module):
            for name, metric in pl_module.metrics.items():
                pl_module.log(name, metric.compute(), prog_bar=True)