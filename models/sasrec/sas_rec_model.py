import torch
import torch.nn as nn

import pytorch_lightning as pl

from losses.sasrec.sas_rec_losses import SASRecBinaryCrossEntropyLoss
from models.bert4rec.bert4rec_model import get_padding_mask
from models.layers.util_layers import MatrixFactorizationLayer, TransformerEmbedding
from configs.sasrec.sas_rec_config import SASRecConfig
from utils.tensor_utils import generate_square_subsequent_mask


class SASRecModel(pl.LightningModule):
    """
    Implementation of the "Self-Attentive Sequential Recommendation" paper.
    see https://doi.org/10.1109%2fICDM.2018.00035 for more details

    see https://github.com/kang205/SASRec for Tensorflow implementation
    """

    def __init__(self, config: SASRecConfig):
        super().__init__()
        self.config = config

        d_model = config.d_model

        self.embedding = TransformerEmbedding(config.item_voc_size, config.max_seq_length, d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model, config.num_transformer_heads, d_model,
                                                    config.transformer_dropout)
        # TODO: check add norm?
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.num_transformer_layers)

        self.input_sequence_mask = None


class SASRecTrainModel(SASRecModel):

    def __init__(self, config: SASRecConfig):
        super().__init__(config)

    def forward(self,
                input_sequence: torch.Tensor,
                pos_items: torch.Tensor,
                neg_items: torch.Tensor,
                position_ids: torch.Tensor = None,
                padding_mask: torch.Tensor = None):
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

        emb_pos_items = self.embedding.get_item_embedding(pos_items)
        emb_neg_items = self.embedding.get_item_embedding(neg_items)

        pos_output = emb_pos_items * transformer_output
        neg_output = emb_neg_items * transformer_output

        pos_output = torch.sum(pos_output, -1)
        neg_output = torch.sum(neg_output, -1)

        return pos_output, neg_output

    def training_step(self, batch, batch_idx, itemizer):
        input_seq = batch['sequence']
        pos = batch['positive_samples']
        neg = batch['negative_samples']

        padding_mask = get_padding_mask(input_seq, itemizer)

        pos_logits, neg_logits = self(input_seq, pos, neg, padding_mask=padding_mask)

        loss_func = SASRecBinaryCrossEntropyLoss()
        loss = loss_func(pos_logits, neg_logits, mask=padding_mask.transpose(0, 1))
        # TODO: add regularization losses

        return {
            'loss': loss
        }


class SASRecTestModel(SASRecModel):

    def __init__(self, config: SASRecConfig):
        super().__init__(config)
