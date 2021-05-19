from typing import Optional, List, Union, Tuple

import torch
import torch.nn as nn

from asme.models.layers.layers import ItemEmbedding
from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.utils.hyperparameter_utils import save_hyperparameters
from data.datasets import USER_ENTRY_NAME


class HGNModel(SequenceRecommenderModel):

    """
    The HGN model for Sequential Recommendation

    The implementation of the paper:
    Chen Ma, Peng Kang, and Xue Liu, "Hierarchical Gating Networks for Sequential Recommendation",
    in the 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2019)
    Arxiv: https://arxiv.org/abs/1906.09217
    """

    @save_hyperparameters
    def __init__(self,
                 user_vocab_size: int,
                 item_vocab_size: int,
                 num_successive_items: int,
                 dims: int,
                 embedding_pooling_type: str = None
                 ):
        """
        inits the HGN model
        :param user_vocab_size: the number of users in the dataset
        :param item_vocab_size: the number of items in the dataset (I)
        :param num_successive_items: the number of successive items
        :param dims: the dimension of the item embeddings (D)
        :param embedding_pooling_type: average or max (seems to be optional?)
        """
        super().__init__()

        # init args
        self.user_vocab_size = user_vocab_size
        self.item_vocab_size = item_vocab_size
        self.dims = dims

        # user and item embeddings
        if self._has_users:
            self.user_embeddings = nn.Embedding(user_vocab_size, dims)
        self.item_embeddings = ItemEmbedding(item_vocab_size, dims, embedding_pooling_type=embedding_pooling_type)

        self.feature_gate_item = nn.Linear(dims, dims)
        self.feature_gate_user = nn.Linear(dims, dims)

        self.instance_gate_item = nn.Parameter(torch.zeros(dims, 1, dtype=torch.float))
        self.instance_gate_user = nn.Parameter(torch.zeros(dims, num_successive_items, dtype=torch.float))

        self.W2 = nn.Embedding(item_vocab_size, dims, padding_idx=0)
        self.b2 = nn.Embedding(item_vocab_size, 1, padding_idx=0)

        # weight initialization
        self.instance_gate_item = nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_user = nn.init.xavier_uniform_(self.instance_gate_user)

        if self._has_users:
            self.user_embeddings.weight.data.normal_(0, 1.0 / dims)
        self.item_embeddings.embedding.weight.data.normal_(0, 1.0 / dims)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

    @property
    def _has_users(self):
        return self.user_vocab_size > 0

    def forward(self,
                sequence: torch.Tensor,
                positive_items: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None,
                negative_items: Optional[torch.Tensor] = None,
                user: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        :param sequence: the sequence :math:'(N, S)'
        :param positive_items: the target items for each sequence :math:'(N, PI)'
        :param padding_mask: the padding mask :math:'(N, S)'
        :param negative_items: the negative items for each sequence :math:'(N, NI)'
        :param user: the user ids for each batch :math:'(N)'
        :return: the logits of the predicted tokens :math:'(N, I)'

        where
            S is the (max) sequence length of the batch,
            N is the batch size,
            D is the dimensions,
            PI is the number of positive (target) samples
            NI is the number of negative samples, and
            I is the number of items
        """
        item_embs = self.item_embeddings(sequence)

        user_provided = user is not None
        if user_provided:
            if not self._has_users:
                raise ValueError("no user voc size specified but users provided")
            user_emb = self.user_embeddings(user)
        else:
            user_emb = None

        # feature gating to select salient latent features of items
        param = self.feature_gate_item(item_embs)
        if user_provided:
            param += self.feature_gate_user(user_emb).unsqueeze(1)  # personalized feature gating
        gate = torch.sigmoid(param)
        gated_item = item_embs * gate  # (N, S, D)

        # instance gating to select the informative items
        secondparam = self.instance_gate_item.unsqueeze(0).squeeze()
        if user_provided:
            secondparam += user_emb.mm(self.instance_gate_user)  # personalized instance gating
        instance_score = torch.sigmoid(torch.matmul(gated_item, secondparam))  # (N, S)
        union_out = gated_item * instance_score.unsqueeze(2)
        union_out = torch.sum(union_out, dim=1)
        union_out = union_out / torch.sum(instance_score, dim=1).unsqueeze(1)  # (N, D)

        positive_item_score = self._calc_scores(positive_items, union_out, item_embs, user_emb)

        if negative_items is None:
            return positive_item_score  # (N, I)

        negative_item_score = self._calc_scores(negative_items, union_out, item_embs, user_emb)

        return positive_item_score, negative_item_score

    def _calc_scores(self,
                     items: torch.Tensor,
                     sequence_representation: torch.Tensor,
                     item_embedding: torch.Tensor,
                     user_emb: torch.Tensor
                     ) -> torch.Tensor:
        w2 = self.W2(items)  # (N, I, D) for train
        b2 = self.b2(items)  # (N, I, 1) for train

        # matrix factorization
        if user_emb is not None:
            res = torch.baddbmm(b2, w2, user_emb.unsqueeze(2)).squeeze()
            # union-level
            res += torch.bmm(sequence_representation.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()
        else:
            res = torch.baddbmm(b2, w2, sequence_representation.unsqueeze(2)).squeeze()  # (N, I)

        # item-item product (to model the relation between two single items)
        rel_score = item_embedding.bmm(w2.permute(0, 2, 1))  # (N, S, I)
        rel_score = torch.sum(rel_score, dim=1)  # (N, I)

        return res + rel_score

    def optional_metadata_keys(self) -> List[str]:
        return [USER_ENTRY_NAME]
