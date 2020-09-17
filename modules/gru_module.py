from typing import Union, List, Dict

import torch

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning import EvalResult
from torch import Tensor

from configs.models.gru.gru_config import GRUConfig
from configs.training.gru.gru_config import GRUTrainingConfig
from metrics.base import AggregatingMetricTrait
from models.gru.gru_model import GRUSeqItemRecommenderModel


class GRUModule(pl.LightningModule):

    def __init__(self,
                 training_config: GRUTrainingConfig,
                 model_config: GRUConfig,
                 metrics: List[AggregatingMetricTrait]
                 ):

        super(GRUModule, self).__init__()

        self.training_config = training_config
        self.model_config = model_config

        self.model = GRUSeqItemRecommenderModel(model_config)

        self._metrics = metrics

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
            lr=self.training_config.learning_rate,
            betas=(self.training_config.beta_1, self.training_config.beta_2)
        )
