from argparse import ArgumentParser
from typing import Optional

import torch
import torch.nn as nn

import pytorch_lightning as pl

from losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss
from models.bert4rec.bert4rec_model import get_padding_mask
from models.layers.util_layers import MatrixFactorizationLayer
from models.layers.transformer_layers import TransformerEmbedding
from configs.sasrec.sas_rec_config import SASRecConfig
from utils.tensor_utils import generate_square_subsequent_mask


class SASRecModel(pl.LightningModule):
    """
    Implementation of the "Self-Attentive Sequential Recommendation" paper.
    see https://doi.org/10.1109%2fICDM.2018.00035 for more details

    see https://github.com/kang205/SASRec for Tensorflow implementation
    """

    SASREC_LEARNING_RATE = 'learning_rate'
    SASREC_ADAM_BETA_1 = 'beta_1'
    SASREC_ADAM_BETA_2 = 'beta_2'

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        # delegate this to the config class
        parser = SASRecConfig.add_model_specific_args(parent_parser)
        parser.add_argument('--{}'.format(cls.SASREC_LEARNING_RATE), default=0.001, type=float,
                            help='the learning rate of the Adam optimizer')
        parser.add_argument('--{}'.format(cls.SASREC_ADAM_BETA_1), default=0.9, type=float,
                            help='the beta 1 of the Adam optimizer')
        parser.add_argument('--{}'.format(cls.SASREC_ADAM_BETA_2), default=0.98, type=float,
                            help='the beta 2 of the Adam optimizer')
        return parser

    def __init__(self, **kwargs):
        """
        inits the SASRec model
        :param kwargs: all arguments added by add_model_specific_args
        """
        super().__init__()

        self.learning_rate = kwargs.get(SASRecModel.SASREC_LEARNING_RATE)
        self.beta1 = kwargs.get(SASRecModel.SASREC_ADAM_BETA_1)
        self.beta2 = kwargs.get(SASRecModel.SASREC_ADAM_BETA_2)

        config = SASRecConfig.from_args(**kwargs)

        self.config = config

        hidden_size = config.d_model
        dropout = config.transformer_dropout

        self.embedding = TransformerEmbedding(config.item_voc_size, config.max_seq_length, hidden_size, dropout)

        encoder_layers = nn.TransformerEncoderLayer(hidden_size, config.num_transformer_heads, hidden_size,
                                                    dropout)
        # TODO: check add norm?
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.num_transformer_layers)

        self.input_sequence_mask = None
        self.mfl = MatrixFactorizationLayer(_weight=self.embedding.get_item_embedding_weight())
        self.softmax = nn.Softmax(dim=2)# TODO: check dim

    def forward(self,
                input_sequence: torch.Tensor,
                pos_items: torch.Tensor,
                neg_items: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None):
        """
        forword pass to generate the scores for the next items in the sequence

        :param input_sequence: the sequence [S x B]
        :param pos_items: TODO
        :param neg_items: TODO
        :param position_ids: the optional position ids [S x B] if not the position ids are generated
        :param padding_mask: the optional padding mask [B x S] if the sequence is padded
        :return: TODO

        Where S is the (max) sequence length of the batch, B the batch size, and I the vocabulary size of the items.
        """
        device = input_sequence.device
        if self.input_sequence_mask is None or self.input_sequence_mask.size(0) != len(input_sequence):
            # to mask the next words we generate a triangle mask
            mask = generate_square_subsequent_mask(len(input_sequence)).to(device)
            self.input_sequence_mask = mask

        input_sequence = self.embedding(input_sequence, position_ids)

        transformer_output = self.transformer_encoder(input_sequence,
                                                      mask=self.input_sequence_mask,
                                                      src_key_padding_mask=padding_mask)
        # when training the model we multiply the seq embedding with the positive and negative items
        if neg_items is not None:
            emb_pos_items = self.embedding.get_item_embedding(pos_items)
            emb_neg_items = self.embedding.get_item_embedding(neg_items)

            pos_output = emb_pos_items * transformer_output
            neg_output = emb_neg_items * transformer_output

            pos_output = torch.sum(pos_output, -1)
            neg_output = torch.sum(neg_output, -1)

            return pos_output, neg_output

        # in test and validation we use the embedding matrix (MF layer) to build softmax for all items
        item_embeddings = self.embedding.get_item_embedding(pos_items)
        output = item_embeddings * transformer_output[0] # use the last step

        output = self.mfl(output)
        return self.softmax(output)

    def training_step(self, batch, batch_idx, itemizer):
        input_seq = batch['sequence']
        pos = batch['positive_samples']
        neg = batch['negative_samples']

        padding_mask = get_padding_mask(input_seq, itemizer)

        pos_logits, neg_logits = self.forward(input_seq, pos, neg_items=neg, padding_mask=padding_mask)

        loss_func = SASRecBinaryCrossEntropyLoss()
        loss = loss_func(pos_logits, neg_logits, mask=padding_mask.transpose(0, 1))
        # TODO: add regularization losses

        return pl.TrainResult(loss)

    def validation_step(self, batch, batch_idx):
        input_seq = batch['sequence']
        # the first entry in each tensor
        items = batch['items']

        scores = self.forward(input_seq, items)

        return pl.EvalResult()

    def configure_optimizers(self):
        # TODO configure learning rate
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
