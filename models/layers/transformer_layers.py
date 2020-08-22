import math

import torch
from torch import nn as nn

from utils.tensor_utils import generate_position_ids


class TransformerEmbedding(nn.Module):
    """
    this transformer embedding combines the item embedding and positional embedding (incl. norm) into a single module
    """

    def __init__(self,
                 item_voc_size: int,
                 max_seq_len: int,
                 embedding_size: int,
                 dropout_probability: float):
        super().__init__()

        self.embedding_size = embedding_size
        self.item_embedding = nn.Embedding(item_voc_size, self.embedding_size)
        self.position_embedding = nn.Embedding(max_seq_len, self.embedding_size)
        self.embedding_norm = nn.LayerNorm(self.embedding_size)
        self.dropout = nn.Dropout(p=dropout_probability)
        # now init the item embedding
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.item_embedding.weight.data.uniform_(- initrange, initrange)

    def get_item_embedding_weight(self) -> torch.Tensor:
        """
        :return: the weight matrix of the item embedding
        some methods reuse the matrix to reduce the parameter size
        """
        return self.item_embedding.weight

    def get_item_embedding(self, input_sequence: torch.Tensor):
        return self.item_embedding(input_sequence)

    def forward(self,
                input_sequence: torch.Tensor,
                position_ids: torch.Tensor = None):
        """
        :param input_sequence: [I x B] where I is the sequence length and B the batch size
        :param position_ids: the position ids [I x B] where I is the sequence length and B the batch size
        :return:
        """
        # generate the position ids if not provided
        input_shape = input_sequence.shape
        seq_length = input_shape[0]
        device = input_sequence.device
        if position_ids is None:
            position_ids = generate_position_ids(seq_length, input_shape, device=device)

        input_sequence = self.item_embedding(input_sequence) * math.sqrt(self.embedding_size)
        input_sequence = input_sequence + self.position_embedding(position_ids)
        input_sequence = self.embedding_norm(input_sequence)

        return input_sequence