from typing import Optional

import torch
import torch.nn as nn

from models.layers.transformer_layers import TransformerEmbedding
from utils.tensor_utils import generate_square_subsequent_mask


class SASRecModel(nn.Module):
    """
    Implementation of the "Self-Attentive Sequential Recommendation" paper.
    see https://doi.org/10.1109%2fICDM.2018.00035 for more details

    see https://github.com/kang205/SASRec for the original Tensorflow implementation
    """

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 dropout: float,
                 embedding_mode: str,
                 ):
        """
        inits the SASRec model
        :param config: all model configurations
        """
        super().__init__()

        self.dropout = dropout
        self.max_seq_length = max_seq_length
        self.transformer_hidden_size = transformer_hidden_size
        self.num_transformer_heads = num_transformer_heads
        self.num_transformer_layers = num_transformer_layers
        self.item_vocab_size = item_vocab_size
        self.embedding_mode = embedding_mode

        self.embedding = TransformerEmbedding(item_voc_size=self.item_vocab_size,
                                              max_seq_len=self.max_seq_length,
                                              embedding_size=self.transformer_hidden_size,
                                              dropout=self.dropout,
                                              embedding_mode=self.embedding_mode)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_hidden_size,
                                                   nhead=self.num_transformer_heads,
                                                   dim_feedforward=self.transformer_hidden_size * 4,
                                                   dropout=self.dropout)
        encoder_norm = nn.LayerNorm(self.transformer_hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                         num_layers=self.num_transformer_layers,
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

        :param input_sequence: the sequence :math:`(S, N)`
        :param pos_items: ids of the positive items (the next items in the sequence) :math:`(N)`
        :param neg_items: random sampled negative items that are not in the session of the user :math:`(N)`
        :param position_ids: the optional position ids if not the position ids are generated :math:`(S, N)`
        :param padding_mask: the optional padding mask if the sequence is padded :math:`(N, S)`
        :return: the logits of the pos_items and the logits of the negative_items, each of shape :math:`(S, N)`
                iff neg_items is provided else the logits for the provided positive items of the same shape

        Where S is the (max) sequence length of the batch and N the batch size.
        """
        # … and H is the hidden size of the transformer/embedding
        device = input_sequence.device
        if self.input_sequence_mask is None or self.input_sequence_mask.size(0) != len(input_sequence):
            # to mask the next words we generate a triangle mask
            mask = generate_square_subsequent_mask(len(input_sequence)).to(device)
            self.input_sequence_mask = mask

        # embed the input sequence
        input_sequence = self.embedding(input_sequence, position_ids)  # (S, N, H)

        # pipe the embedded sequence to the transformer
        transformer_output = self.transformer_encoder(input_sequence,
                                                      mask=self.input_sequence_mask,
                                                      src_key_padding_mask=padding_mask)  # (S, N, H)
        # when training the model we multiply the seq embedding with the positive and negative items
        if neg_items is not None:
            emb_pos_items = self.embedding.get_item_embedding(pos_items)  # (N, H)
            emb_neg_items = self.embedding.get_item_embedding(neg_items)  # (N, H)

            pos_output = emb_pos_items * transformer_output  # (S, N, H)
            neg_output = emb_neg_items * transformer_output  # (S, N, H)

            pos_output = torch.sum(pos_output, -1)  # (S, N)
            neg_output = torch.sum(neg_output, -1)  # (S, N)

            return pos_output, neg_output

        # inference step (I is the number of positive items to test)
        # embeddings of pos_items
        item_embeddings = self.embedding.get_item_embedding(pos_items, flatten=False)  # (I, N, H)

        # permute embeddings for batch matrix multiplication
        item_embeddings = item_embeddings.permute(1, 2, 0)  # (N, H, I)
        transformer_output = transformer_output.transpose(0, 1)  # (N, S, H)

        output = torch.matmul(transformer_output, item_embeddings)  # (N, S, I)

        # we use "advanced" indexing to slice the right elements from the sequence.
        batch_size = output.size()[0]
        batch_index = torch.arange(0, batch_size)

        # calculate indices from the padding mask
        seq_index = (~padding_mask).sum(-1) - 1
        scores = output[batch_index, seq_index, :]  # (N, I)
        return scores.transpose(0, 1)  # (I, N), to be consistent with the input format
