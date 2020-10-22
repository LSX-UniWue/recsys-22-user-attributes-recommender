from typing import Tuple

import torch


from torch import nn

from configs.models.caser.caser_config import CaserConfig
from models.layers.util_layers import get_activation_layer


class CaserModel(nn.Module):
    """
        implementation of the Caser model proposed in
        Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang, WSDM'18
        see https://doi.org/10.1145/3159652.3159656 for more details

        original pytorch implementation: https://github.com/graytowne/caser_pytorch
    """

    @classmethod
    def from_config(cls,
                    config: CaserConfig) -> 'CaserModel':
        return cls(embedding_size=config.d,
                   item_voc_size=config.item_voc_size,
                   user_voc_size=config.user_voc_size,
                   max_seq_length=config.max_seq_length,
                   num_vertical_filters=config.num_vertical_filters,
                   num_horizontal_filters=config.num_horizontal_filters,
                   conv_activation_fn=config.conv_activation_fn,
                   fc_activation_fn=config.fc_activation_fn,
                   dropout=config.dropout)

    def __init__(self,
                 embedding_size: int,
                 item_voc_size: int,
                 user_voc_size: int,
                 max_seq_length: int,
                 num_vertical_filters: int,
                 num_horizontal_filters: int,
                 conv_activation_fn: str,
                 fc_activation_fn: str,
                 dropout: float
                 ):
        super().__init__()

        self.embedding_size = embedding_size
        self.item_voc_size = item_voc_size
        self.user_voc_size = user_voc_size
        self.max_seq_length = max_seq_length
        self.num_vertical_filters = num_vertical_filters
        self.num_horizontal_filters = num_horizontal_filters
        self.conv_activation_fn = conv_activation_fn
        self.fc_activation_fn = fc_activation_fn
        self.dropout = dropout

        # user and item embedding
        if self._has_users:
            self.user_embedding = nn.Embedding(user_voc_size, embedding_dim=self.embedding_size)
        self.item_embedding = nn.Embedding(item_voc_size, embedding_dim=self.embedding_size)

        # vertical conv layer
        self.conv_vertical = nn.Conv2d(1, num_vertical_filters, (self.max_seq_length, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_length)]
        self.conv_horizontal = nn.ModuleList(
            [CaserHorizontalConvNet(1, num_vertical_filters, (length, embedding_size), self.conv_activation_fn)
                for length in lengths]
        )

        # fully-connected layer
        self.fc1_dim_vertical = num_vertical_filters * embedding_size
        fc1_dim_horizontal = num_horizontal_filters * len(lengths)
        fc1_dim = self.fc1_dim_vertical + fc1_dim_horizontal

        self.fc1 = nn.Linear(fc1_dim, embedding_size)

        self.W2 = nn.Embedding(item_voc_size, 2 * embedding_size if self._has_users else embedding_size)
        self.b2 = nn.Embedding(item_voc_size, 1)

        self.fc1_activation = get_activation_layer(self.fc_activation_fn)

        # dropout
        self.dropout = nn.Dropout(self.dropout)

        self._init_weights()

    @property
    def _has_users(self):
        return self.user_voc_size > 0

    def _init_weights(self):
        # weight initialization
        if self._has_users:
            self.user_embedding.weight.data.normal_(0, 1.0 / self.user_embedding.embedding_dim)

        self.item_embedding.weight.data.normal_(0, 1.0 / self.item_embedding.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None

    def forward(self,
                sequences: torch.Tensor,
                users: torch.Tensor,
                pos_items: torch.Tensor,
                neg_items: torch.Tensor):
        """
        forward pass for the
        :param sequences: the sequences [B x S]
        :param users: the users for each batch [B]
        :param pos_items: the positive (next) items of the sequence
        :param neg_items: the negative items (sampled)
        :return:

        Where B is the batch size, S the max sequence length (of the current batch),
        """
        # Embedding Look-up
        item_embs = self.item_embedding(sequences).unsqueeze(1)  # use unsqueeze() to get 4-D

        users_provided = users is not None
        if users_provided:
            if not self._has_users:
                raise ValueError("no user voc size specified but users provided")
            user_emb = self.user_embedding(users).squeeze(1)
        else:
            user_emb = None

        # Convolutional Layers
        out_vertical = None
        # vertical conv layer
        if self.config.num_vertical_filters > 0:
            out_vertical = self.conv_vertical(item_embs)
            out_vertical = out_vertical.view(-1, self.fc1_dim_vertical)  # prepare for fully connect

        # horizontal conv layer
        out_horizontal = None
        out_hs = list()
        if self.config.num_horizontal_filters > 0:
            for conv in self.conv_horizontal:
                conv_out = conv(item_embs)
                out_hs.append(conv_out)
            out_horizontal = torch.cat(out_hs, 1)

        # Fully-connected Layers
        out = torch.cat([out_vertical, out_horizontal], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.fc1_activation(self.fc1(out))
        x = torch.cat([z, user_emb], 1) if users_provided else z

        pos_w2 = self.W2(pos_items)
        pos_b2 = self.b2(pos_items)

        if not self.training:
            w2 = pos_w2.squeeze()
            b2 = pos_b2.squeeze()
            return (x * w2).sum(1) + b2

        x_unsqueezed = x.unsqueeze(2)
        res_pos = torch.baddbmm(pos_b2, pos_w2, x_unsqueezed).squeeze()

        # negative items
        neg_w2 = self.W2(neg_items)
        neg_b2 = self.b2(neg_items)
        res_negative = torch.baddbmm(neg_b2, neg_w2, x_unsqueezed).squeeze()
        return res_pos, res_negative


class CaserHorizontalConvNet(nn.Module):
    """
    the horizontal convolution module for the Caser model
    """
    def __init__(self,
                 max_length: int,
                 num_filters: int,
                 kernel_size: Tuple[int, int],
                 activation_fn: str
                 ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        self.conv_activation = get_activation_layer(activation_fn)
        length = kernel_size[0]
        self.conv_pooling = nn.MaxPool1d(kernel_size=max_length - length)

    def forward(self,
                input_tensor: torch.Tensor):
        conv_out = self.conv(input_tensor).squeeze(3)
        conv_out = self.conv_activation(conv_out)
        return self.conv_pooling(conv_out).squeeze(2)
