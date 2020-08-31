import torch

import pytorch_lightning as pl

from configs.training.sasrec.sas_rec_config import SASRecTrainingConfig
from losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss
from modules.bert4rec_module import get_padding_mask
from configs.models.sasrec.sas_rec_config import SASRecConfig
from models.sasrec.sas_rec_model import SASRecModel
from registry import registry


@registry.register_module('sasrec')
class SASRecModule(pl.LightningModule):

    def __init__(self,
                 training_config: SASRecTrainingConfig,
                 model_config: SASRecConfig
                 ):
        """
        inits the SASRec module
        :param training_config: all training configurations
        :param model_config: all model configurations
        """
        super().__init__()

        self.training_config = training_config

        self.model_config = model_config
        self.model = SASRecModel(self.model_config)

    def training_step(self, batch, batch_idx, itemizer):
        input_seq = batch['sequence']
        pos = batch['positive_samples']
        neg = batch['negative_samples']

        padding_mask = get_padding_mask(input_seq, itemizer)

        pos_logits, neg_logits = self.model(input_seq, pos, neg_items=neg, padding_mask=padding_mask)

        loss_func = SASRecBinaryCrossEntropyLoss()
        loss = loss_func(pos_logits, neg_logits, mask=padding_mask.transpose(0, 1))
        # the original code
        # (https://github.com/kang205/SASRec/blob/641c378fcfac265ea8d1e5fe51d4d53eb892d1b4/model.py#L92)
        # adds regularization losses, but they are empty, as far as I can see (dzo)
        # TODO: check

        return pl.TrainResult(loss)

    def validation_step(self, batch, batch_idx):
        input_seq = batch['sequence']
        # the first entry in each tensor
        items = batch['items']
        scores = self.forward(input_seq, items)

        return pl.EvalResult()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.training_config.learning_rate,
                                betas=(self.training_config.beta1, self.training_config.beta2))
