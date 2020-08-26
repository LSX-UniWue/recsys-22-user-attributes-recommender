from argparse import ArgumentParser
from typing import Optional

import torch
import torch.nn as nn

import pytorch_lightning as pl

from losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss
from models.bert4rec.bert4rec_model import get_padding_mask
from models.layers.transformer_layers import TransformerEmbedding
from configs.models.sasrec.sas_rec_config import SASRecConfig
from utils.tensor_utils import generate_square_subsequent_mask


class SASRecModel(pl.LightningModule):
    """
    Implementation of the "Self-Attentive Sequential Recommendation" paper.
    see https://doi.org/10.1109%2fICDM.2018.00035 for more details

    see https://github.com/kang205/SASRec for the original Tensorflow implementation
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

        self.config = SASRecConfig.from_args(kwargs)

        hidden_size = self.config.transformer_hidden_size
        dropout = self.config.transformer_dropout

        self.embedding = TransformerEmbedding(item_voc_size=self.config.item_voc_size,
                                              max_seq_len=self.config.max_seq_length,
                                              embedding_size=hidden_size,
                                              dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=self.config.num_transformer_heads,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout)
        encoder_norm = nn.LayerNorm(hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                         num_layers=self.config.num_transformer_layers,
                                                         norm=encoder_norm)

        self.input_sequence_mask = None

    def forward(self,
                input_sequence: torch.Tensor,
                pos_items: torch.Tensor,
                neg_items: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None):
        """
        Forward pass to generate the logits for the positive (next) items and the negative (randomly sampled items,
        that are not in the current sequence) items.
        If no negative items are provided,

        :param input_sequence: the sequence [S x B]
        :param pos_items: ids of the positive items (the next items in the sequence)
        :param neg_items: random sampled negative items that are not in the session of the user
        :param position_ids: the optional position ids [S x B] if not the position ids are generated
        :param padding_mask: the optional padding mask [B x S] if the sequence is padded
        :return: the logits of the pos_items and the logits of the negative_items.


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

        # inference step
        # TODO: check, strange reshaping
        # original code
        # seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])
        # self.test_item = tf.placeholder(tf.int32, shape=(101))
        # test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        # self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        # self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, 101])
        # self.test_logits = self.test_logits[:, -1, :]
        item_embeddings = self.embedding.get_item_embedding(pos_items)
        item_embeddings = item_embeddings.permute(1, 2, 0)
        transformer_output = transformer_output.transpose(0, 1)
        output = torch.matmul(transformer_output, item_embeddings)

        # here we return the logits of the last step of the sequence
        output = output[:, -1, :]
        return output.transpose(0, 1)

    def training_step(self, batch, batch_idx, itemizer):
        input_seq = batch['sequence']
        pos = batch['positive_samples']
        neg = batch['negative_samples']

        padding_mask = get_padding_mask(input_seq, itemizer)

        pos_logits, neg_logits = self.forward(input_seq, pos, neg_items=neg, padding_mask=padding_mask)

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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.beta1, self.beta2))
