import torch

import pytorch_lightning as pl

from configs.training.sasrec.sas_rec_config import SASRecTrainingConfig
from losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss
from modules.bert4rec_module import get_padding_mask
from configs.models.sasrec.sas_rec_config import SASRecConfig
from models.sasrec.sas_rec_model import SASRecModel
from registry import registry
from tokenization.tokenizer import Tokenizer




class SASRecModule(pl.LightningModule):

    def __init__(self,
                 training_config: SASRecTrainingConfig,
                 model_config: SASRecConfig,
                 tokenizer: Tokenizer,
                 batch_first: bool = True
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
        self.tokenizer = tokenizer
        self.batch_first = batch_first

    def training_step(self, batch, batch_idx):
        input_seq = batch['session']
        pos = batch['positive_samples']
        neg = batch['negative_samples']

        if self.batch_first:
            input_seq = input_seq.transpose(1, 0)
            pos = pos.transpose(1, 0)
            neg = neg.transpose(1, 0)

        padding_mask = get_padding_mask(input_seq, self.tokenizer)

        pos_logits, neg_logits = self.model(input_seq, pos, neg_items=neg, padding_mask=padding_mask)

        loss_func = SASRecBinaryCrossEntropyLoss()
        loss = loss_func(pos_logits, neg_logits, mask=padding_mask.transpose(0, 1))
        # the original code
        # (https://github.com/kang205/SASRec/blob/641c378fcfac265ea8d1e5fe51d4d53eb892d1b4/model.py#L92)
        # adds regularization losses, but they are empty, as far as I can see (dzo)
        # TODO: check

        return pl.TrainResult(loss)

    def validation_step(self, batch, batch_idx):
        input_seq = batch['session']
        # the first entry in each tensor
        #items = batch['items']
        #scores = self.forward(input_seq, items)

        return pl.EvalResult()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.training_config.learning_rate,
                                betas=(self.training_config.beta_1, self.training_config.beta_2))
