import torch

import pytorch_lightning as pl
import torch.nn as nn

from configs.models.gru.gru_config import GRUConfig
from configs.training.gru.gru_config import GRUTrainingConfig
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

    def forward(self, session, lengths, batch_idx):
        return self.model.forward(session, lengths, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.learning_rate,
            betas=(self.training_config.beta_1, self.training_config.beta_2)
        )
