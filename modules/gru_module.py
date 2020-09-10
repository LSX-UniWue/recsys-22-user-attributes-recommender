from typing import Union, List, Dict

import torch

import pytorch_lightning as pl
import torch.nn as nn
from torch import Tensor

from configs.models.gru.gru_config import GRUConfig
from configs.training.gru.gru_config import GRUTrainingConfig
from metrics.ranking_metrics import RecallAtMetric
from models.gru.gru_model import GRUSeqItemRecommenderModel


class GRUModule(pl.LightningModule):

    def __init__(self,
                 training_config: GRUTrainingConfig,
                 model_config: GRUConfig):
        super(GRUModule, self).__init__()

        self.training_config = training_config
        self.model_config = model_config

        self.model = GRUSeqItemRecommenderModel(model_config)

    def training_step(self, batch, batch_idx):
        prediction = self.forward(batch["session"], batch["session_length"], batch_idx)
        # TODO: discuss: maybe the target should be generated here and not in the dataset; see bert4rec

        loss = nn.CrossEntropyLoss()(prediction, batch["target"])

        return {
            "loss": loss
        }

    # FIXME need to include loss value in validation result otherwise no checkpoints will be saved
    def validation_step(self, batch, batch_idx):
        result = pl.EvalResult()

        prediction = self.forward(batch["session"], batch["session_length"], batch_idx)

        recall_at_metric = RecallAtMetric(k=5)
        tp, tpfn, recall_at_k = recall_at_metric(prediction, batch["target"])

        result.tp = tp
        result.tpfn = tpfn

        return result

    def validation_epoch_end(self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]) -> Dict[
        str, Dict[str, Tensor]]:
        result = pl.EvalResult()
        result.log("recall_at_k", outputs["tp"].sum() / outputs["tpfn"].sum(), prog_bar=True)

        return result

    def forward(self, session, lengths, batch_idx):
        return self.model.forward(session, lengths, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.learning_rate,
            betas=(self.training_config.beta_1, self.training_config.beta_2)
        )
