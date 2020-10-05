from typing import Union, List, Dict

import torch

import pytorch_lightning as pl
import torch.nn as nn
from pyhocon import ConfigTree
from pytorch_lightning import EvalResult
from torch import Tensor

from metrics.ranking_metrics import RecallAtMetric, MRRAtMetric
from models.gru.gru_model import GRUSeqItemRecommenderModel


class GRUModule(pl.LightningModule):

    def __init__(self,
                 model: GRUSeqItemRecommenderModel,
                 lr: float,
                 beta_1: float,
                 beta_2: float,
                 enable_metrics: bool,
                 metrics_k: List[int]
                 ):

        super(GRUModule, self).__init__()

        self.model = model
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self._metrics = []

        if enable_metrics:
            self._metrics = [RecallAtMetric(k) for k in metrics_k] + [MRRAtMetric(k) for k in metrics_k]

    @staticmethod
    def from_configuration(config: ConfigTree) -> 'GRUModule':
        model = GRUSeqItemRecommenderModel.from_configuration(config)

        optimizer_config = config["optimizer"]
        lr = optimizer_config.get_float("learning_rate")
        beta_1 = optimizer_config.get_float("beta_1")
        beta_2 = optimizer_config.get_float("beta_2")

        enable_metrics = config.get_bool("metrics.enable_metrics", False)
        metrics_k = config.get_list("metrics.k", [])

        return GRUModule(model, lr, beta_1, beta_2, enable_metrics, metrics_k)

    def training_step(self, batch, batch_idx):
        prediction = self._forward(batch["session"], batch["session_length"], batch_idx)
        # TODO: discuss: maybe the target should be generated here and not in the dataset; see bert4rec

        loss = self.loss(prediction, batch["target"])

        return {
            "loss": loss
        }

    def loss(self, prediction, target):
        return nn.CrossEntropyLoss()(prediction, target)

    def validation_step(self, batch, batch_idx):

        prediction = self._forward(batch["session"], batch["session_length"], batch_idx)

        target = batch["target"]

        loss = self.loss(prediction, target)
        result = EvalResult(checkpoint_on=loss)
        result.log("val_loss", loss, on_step=True, on_epoch=True)

        for metric in self._metrics:
            metric.on_step_end(prediction, target, result)

        return result

    def validation_epoch_end(self, outputs: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]) -> Dict[str, Dict[str, Tensor]]:

        result = EvalResult(checkpoint_on=outputs["step_val_loss"].mean())

        for metric in self._metrics:
            metric.on_epoch_end(outputs, result)

        return result

    def test_step(self, batch, batch_idx) -> EvalResult:
        prediction = self._forward(batch["session"], batch["session_length"], batch_idx)

        result = EvalResult()

        for metric in self._metrics:
            metric.on_step_end(prediction, batch["target"], result)

        return result

    def test_epoch_end(self, outputs: Union[Dict[str, Tensor], List[Dict[str, Tensor]]]) -> EvalResult:
        result = EvalResult()

        for metric in self._metrics:
            metric.on_epoch_end(outputs, result)

        return result

    def _forward(self, session, lengths, batch_idx):
        return self.model(session, lengths, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta_1, self.beta_2)
        )
