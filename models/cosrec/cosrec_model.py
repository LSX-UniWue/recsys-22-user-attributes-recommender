import torch
import torch.nn as nn
from typing import List

from models.layers.layers import ItemEmbedding
from models.layers.util_layers import get_activation_layer


class CosRecModel(nn.Module):
    """
    A 2D CNN for sequential Recommendation.

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

        # build cnnBlock
        self.block_num = block_num
        block_dim.insert(0, 2 * embed_dim)  # adds first input dimension of first cnn block
        self.cnnBlock = [0] * block_num
        for i in range(block_num):
            self.cnnBlock[i] = CNNBlock(block_dim[i], block_dim[i + 1])
        self.cnnBlock = nn.ModuleList(self.cnnBlock)  # holds submodules in a list
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / embed_dim)
        self.item_embeddings.get_weight().data.normal_(0, 1.0 / embed_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    def forward(self,
                sequence: torch.Tensor,
                user: torch.Tensor,
                item_to_predict: torch.Tensor,
                eval: bool = False
                ):
        """
        Args:
            sequence: torch.Tensor with size :math`(N, S)`
                a batch of sequence
            user: torch.Tensor with size :math`(N)`
                a batch of user
            item_to_predict: torch.Tensor with size :math`(N)`
                a batch of items
            eval: boolean, optional
                Train or Prediction. Set to True when evaluation.

            where N is the batch size and S the max sequence length
        """

        item_embs = self.item_embeddings(sequence)  # (N, S, D)

        # cast all item embeddings pairs against each other
        item_i = torch.unsqueeze(item_embs, 1)  # (N, 1, S, D)
        item_i = item_i.repeat(1, self.seq_len, 1, 1)  # (N, S, S, D)
        item_j = torch.unsqueeze(item_embs, 2)  # (N, S, 1, D)
        item_j = item_j.repeat(1, 1, self.seq_len, 1)  # (N, S, S, D)
        all_embed = torch.cat([item_i, item_j], 3)  # (N, S, S, 2*D)
        out = all_embed.permute(0, 3, 1, 2)

        # 2D CNN
        mb = sequence.shape[0]
        for i in range(self.block_num):
            out = self.cnnBlock[i](out)  # forward CNN blocks
        out = self.avg_pool(out).reshape(mb, self.cnn_out_dim)
        out = out.squeeze(-1).squeeze(-1)

        # apply fc and dropout
        out = self.activation_function(self.fc1(out))
        out = self.dropout(out)

        if user is not None:
            user_emb = self.user_embeddings(user)  # (N, 1, D)

            x = torch.cat([out, user_emb.squeeze(1)], 1)
        else:
            x = out

        w2 = self.W2(item_to_predict)
        b2 = self.b2(item_to_predict)
        if eval:
            w2 = w2.squeeze()       # removed: b2 = b2.squeeze
            return (x * w2).sum(1) + b2

        return torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()


class CNNBlock(nn.Module):
    """
    CNN block with two layers.

    For this illustrative model, we don't change stride and padding.
    But to design deeper models, some modifications might be needed.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.bn2 = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out
