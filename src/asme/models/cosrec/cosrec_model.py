import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from asme.models.layers.layers import ItemEmbedding
from asme.models.layers.util_layers import get_activation_layer
from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from data.datasets import USER_ENTRY_NAME


class CosRecModel(SequenceRecommenderModel):
    """
    A 2D CNN for sequential Recommendation.
    Based on paper "CosRec: 2D Convolutional Neural Networks for Sequential Recommendation" which can be found at
    https://dl.acm.org/doi/10.1145/3357384.3358113.
    Original code used for this model is available at: https://github.com/zzxslp/CosRec.

    Args:
        user_vocab_size: number of users.
        item_vocab_size: number of items.
        max_seq_length: length of sequence, Markov order.
        embed_dim: dimensions for user and item embeddings. (latent dimension in paper)
        block_num: number of cnn blocks. (convolutional layers??)
        block_dim: the dimensions for each block. len(block_dim)==block_num -> List
        fc_dim: dimension of the first fc layer, mainly for dimension reduction after CNN.
        activation_function: type of activation functions (string) to use for the output fcn
        dropout: dropout ratio.
    """

    def __init__(self,
                 user_vocab_size: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 embed_dim: int,
                 block_num: int,
                 block_dim: List[int],
                 fc_dim: int,
                 activation_function: str,
                 dropout: float,
                 embedding_pooling_type: str = None
                 ):
        super().__init__()

        assert len(block_dim) == block_num

        self.seq_len = max_seq_length
        self.embed_dim = embed_dim
        self.cnn_out_dim = block_dim[-1]  # dimension of output of last cnn block
        self.fc_dim = fc_dim

        # user and item embeddings
        self.user_embeddings = nn.Embedding(user_vocab_size, embed_dim)
        self.item_embeddings = ItemEmbedding(item_vocab_size, embed_dim, embedding_pooling_type=embedding_pooling_type)

        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items
        dim = self.fc_dim + embed_dim if user_vocab_size > 0 else self.fc_dim
        self.W2 = nn.Embedding(item_vocab_size, dim)
        self.b2 = nn.Embedding(item_vocab_size, 1)

        # dropout and fc layer
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.cnn_out_dim, self.fc_dim)
        self.activation_function = get_activation_layer(activation_function)

        # build cnn block
        self.block_num = block_num
        block_dim.insert(0, 2 * embed_dim)  # adds first input dimension of first cnn block
        # holds submodules in a list
        self.cnnBlock = nn.ModuleList(CNNBlock(block_dim[i], block_dim[i + 1]) for i in range(block_num))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / embed_dim)
        self.item_embeddings.get_weight().data.normal_(0, 1.0 / embed_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self,
                sequence: torch.Tensor,
                positive_items: torch.Tensor,
                negative_items: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None,
                user: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            sequence: torch.Tensor with size :math`(N, S)`
                a batch of sequence
            user: torch.Tensor with size :math`(N)`
                a batch of user
            padding_mask: torch.Tensor with size :math`(N, S)` note: not used
            positive_items: torch.Tensor with size :math`(N, PI)`
                a batch of items
            negative_items: torch.Tensor with size :math`(N, NI)`
            eval: boolean, optional
                Train or Prediction. Set to True when evaluation.

            where N is the batch size, S the max sequence length, PI the positive item size, NI the negative item size
        """

        item_embs = self.item_embeddings(sequence)  # (N, S, D)
        sequence_shape = sequence.size()
        batch_size = sequence_shape[0]
        max_sequence_length = sequence_shape[1]
        # cast all item embeddings pairs against each other
        item_i = torch.unsqueeze(item_embs, 1)  # (N, 1, S, D)
        item_i = item_i.repeat(1, max_sequence_length, 1, 1)  # (N, S, S, D)
        item_j = torch.unsqueeze(item_embs, 2)  # (N, S, 1, D)
        item_j = item_j.repeat(1, 1, max_sequence_length, 1)  # (N, S, S, D)
        all_embed = torch.cat([item_i, item_j], 3)  # (N, S, S, 2*D)
        out = all_embed.permute(0, 3, 1, 2)  # (N, 2 * D, S, S)

        # 2D CNN
        for i in range(self.block_num):
            out = self.cnnBlock[i](out)

        out = self.avg_pool(out).reshape(batch_size, self.cnn_out_dim)  # (N, C_O)
        out = out.squeeze(-1).squeeze(-1)

        # apply fc and dropout
        out = self.activation_function(self.fc1(out))  # (N, F_D)
        out = self.dropout(out)

        if user is not None:
            user_emb = self.user_embeddings(user)  # (N, D)
            x = torch.cat([out, user_emb], 1)  # (N, 2 * D)
        else:
            x = out

        w2 = self.W2(positive_items)  # (N, I, F_D)
        b2 = self.b2(positive_items)  # (N, I, 1)

        x = x.unsqueeze(2)
        # TODO: use torch.einsum('nif,nf -> ni', w2, x) + b2.squeeze()
        positive_score = torch.baddbmm(b2, w2, x).squeeze()

        if negative_items is None:
            return positive_score

        w2_negative_items = self.W2(negative_items)
        b2_negative_items = self.b2(negative_items)

        negative_score = torch.baddbmm(b2_negative_items, w2_negative_items, x).squeeze()

        return positive_score, negative_score

    def optional_metadata_keys(self) -> List[str]:
        return [USER_ENTRY_NAME]


class CNNBlock(nn.Module):
    """
    CNN block with two layers.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int
                 ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm2d(output_dim)
        self.batch_norm2 = nn.BatchNorm2d(output_dim)

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        out = self.relu(self.batch_norm1(self.conv1(x)))
        return self.relu(self.batch_norm2(self.conv2(out)))

