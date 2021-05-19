from typing import Tuple, Optional, Union, List

import torch

from torch import nn

from asme.models.layers.layers import ItemEmbedding
from asme.models.layers.util_layers import get_activation_layer
from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.utils.hyperparameter_utils import save_hyperparameters
from data.datasets import USER_ENTRY_NAME


class CaserModel(SequenceRecommenderModel):
    """
        implementation of the Caser model proposed in
        Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding, Jiaxi Tang and Ke Wang, WSDM'18
        see https://doi.org/10.1145/3159652.3159656 for more details

        adapted from the original pytorch implementation: https://github.com/graytowne/caser_pytorch
    """

    @save_hyperparameters
    def __init__(self,
                 embedding_size: int,
                 item_vocab_size: int,
                 user_vocab_size: int,
                 max_seq_length: int,
                 num_vertical_filters: int,
                 num_horizontal_filters: int,
                 conv_activation_fn: str,
                 fc_activation_fn: str,
                 dropout: float,
                 embedding_pooling_type: str = None
                 ):
        super().__init__()

        self.embedding_size = embedding_size
        self.embedding_pooling_type = embedding_pooling_type
        self.item_vocab_size = item_vocab_size
        self.user_vocab_size = user_vocab_size
        self.max_seq_length = max_seq_length
        self.num_vertical_filters = num_vertical_filters
        self.num_horizontal_filters = num_horizontal_filters
        self.conv_activation_fn = conv_activation_fn
        self.fc_activation_fn = fc_activation_fn
        self.dropout = dropout

        # user and item embedding
        if self._has_users:
            self.user_embedding = nn.Embedding(self.user_vocab_size, embedding_dim=self.embedding_size)
        else:
            print('user vocab size is 0; no user information will be used for training')

        self.item_embedding = ItemEmbedding(item_voc_size=self.item_vocab_size,
                                            embedding_size=self.embedding_size,
                                            embedding_pooling_type=self.embedding_pooling_type)

        # vertical conv layer
        self.conv_vertical = nn.Conv2d(1, self.num_vertical_filters, (self.max_seq_length, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(self.max_seq_length)]

        self.conv_horizontal = nn.ModuleList(
            [CaserHorizontalConvNet(num_filters=self.num_horizontal_filters,
                                    kernel_size=(length, embedding_size),
                                    activation_fn=self.conv_activation_fn,
                                    max_length=self.max_seq_length)
                for length in lengths]
        )

        # fully-connected layer
        self.fc1_dim_vertical = self.num_vertical_filters * self.embedding_size
        fc1_dim_horizontal = self.num_horizontal_filters * len(lengths)
        fc1_dim = self.fc1_dim_vertical + fc1_dim_horizontal

        self.fc1 = nn.Linear(fc1_dim, self.embedding_size)

        self.W2 = nn.Embedding(item_vocab_size, 2 * self.embedding_size if self._has_users else self.embedding_size)
        self.b2 = nn.Embedding(item_vocab_size, 1)

        self.fc1_activation = get_activation_layer(self.fc_activation_fn)

        # dropout
        self.dropout = nn.Dropout(self.dropout)

        self._init_weights()

    @property
    def _has_users(self):
        return self.user_vocab_size > 0

    def _init_weights(self):
        # weight initialization
        # note: item embedding already init when initialized
        if self._has_users:
            self.user_embedding.weight.data.normal_(0, 1.0 / self.user_embedding.embedding_dim)

        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None

    def forward(self,
                sequence: torch.Tensor,
                positive_items: torch.Tensor,
                negative_items: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                user: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        forward pass
        :param padding_mask: the padding mask; note: not used
        :param sequence: the sequences :math`(N, S)`
        :param user: the users for each batch :math `(N)` optional
        :param positive_items: the positive (next) items of the sequence `(N)`
        :param negative_items: the negative items (sampled) `(N, X)`, only required for training
        :return: the logits of the pos_items and if provided the logits of the neg_items

        Where B is the batch size, S the max sequence length (of the current batch)
        """
        # Embedding Look-up
        item_embs = self.item_embedding(sequence).unsqueeze(1)  # use unsqueeze() to get 4-D

        users_provided = user is not None
        if users_provided:
            if not self._has_users:
                raise ValueError("no user voc size specified but users provided")
            user_emb = self.user_embedding(user).squeeze(1)
        else:
            user_emb = None

        # Convolutional Layers
        out_vertical = None
        # vertical conv layer
        if self.num_vertical_filters > 0:
            out_vertical = self.conv_vertical(item_embs)
            out_vertical = out_vertical.view(-1, self.fc1_dim_vertical)  # prepare for fully connect

        # horizontal conv layer
        out_horizontal = None
        out_hs = list()
        if self.num_horizontal_filters > 0:
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

        pos_w2 = self.W2(positive_items)
        pos_b2 = self.b2(positive_items)

        #if not self.training:
        #    w2 = pos_w2.squeeze()
        #    b2 = pos_b2.squeeze()
        #    w2 = w2.permute(1, 0, 2)
        #    return torch.matmul(x, w2).sum(dim=1) + b2

        x = x.unsqueeze(2)
        res_pos = torch.baddbmm(pos_b2, pos_w2, x).squeeze()
        if negative_items is None:
            return res_pos

        # negative items
        neg_w2 = self.W2(negative_items)
        neg_b2 = self.b2(negative_items)
        res_negative = torch.baddbmm(neg_b2, neg_w2, x).squeeze()
        return res_pos, res_negative

    def optional_metadata_keys(self) -> List[str]:
        return [USER_ENTRY_NAME]


class CaserHorizontalConvNet(nn.Module):
    """
    the horizontal convolution module for the Caser model
    """
    def __init__(self,
                 num_filters: int,
                 kernel_size: Tuple[int, int],
                 activation_fn: str,
                 max_length: int
                 ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        self.conv_activation = get_activation_layer(activation_fn)
        length = kernel_size[0]
        self.conv_pooling = nn.MaxPool1d(kernel_size=max_length + 1 - length)

    def forward(self,
                input_tensor: torch.Tensor):
        conv_out = self.conv(input_tensor).squeeze(3)
        conv_out = self.conv_activation(conv_out)
        return self.conv_pooling(conv_out).squeeze(2)
