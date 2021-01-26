from typing import Callable

import torch
from pytorch_lightning import Callback


class LossLoggerCallback(Callback):

    def __init__(self, loss_name: str, loss_fn: Callable[[torch.Tensor], torch.Tensor]):
        self.loss_name = loss_name
        self.loss_fn = loss_fn

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pl_module.log(self.loss_name, self.loss_fn(batch))