import torch
import torch.nn as nn

from asme.models.layers.layers import ItemEmbedding
from asme.utils.hyperparameter_utils import save_hyperparameters


class HGNModel(nn.Module):

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
                user: torch.Tensor,
                items_to_predict: torch.Tensor,
                for_pred: bool = False
                ) -> torch.Tensor:
        """
        Forward pass

        :param sequence: the sequence :math:'(N, S)'
        :param user: the user ids for each batch :math:'(N)'
        :param items_to_predict: the target items for each sequence :math:'(N, I)'
        :param for_pred: true if logits are used for prediction
        :return: the logits of the predicted tokens :math:'(N, I)'

        where
            S is the (max) sequence length of the batch,
            N is the batch size,
            D is the dimensions, and
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

        w2 = self.W2(items_to_predict)  # (N, I, D) for train, (I, D) for prediction
        b2 = self.b2(items_to_predict)  # (N, I, 1) for train, (I, 1) for prediction

        if for_pred:
            w2 = w2.squeeze()  # (I, D)
            b2 = b2.squeeze()  # (I)

            # Matrix Factorization
            if user_provided:
                res = user_emb.mm(w2.t()) + b2  # (N, I)
                # union-level
                res += union_out.mm(w2.t())  # (N, I)
            else:
                res = union_out.mm(w2.t())  # (N, I)

            # item-item product (to model the relation between two single items)
            rel_score = torch.matmul(item_embs, w2.t().unsqueeze(0))  # (B, L?, I)
            rel_score = torch.sum(rel_score, dim=1)  # (B, I)
            res += rel_score # (B, I)
        else:
            # Matrix Factorization
            if user_provided:
                res = torch.baddbmm(b2, w2, user_emb.unsqueeze(2)).squeeze()
                # union-level
                res += torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()
            else:
                res = torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()  # (N, I)

            # item-item product (to model the relation between two single items)
            rel_score = item_embs.bmm(w2.permute(0, 2, 1))  # (N, S, I)
            rel_score = torch.sum(rel_score, dim=1)  # (N, I)
            res += rel_score  # (N, I)
        return res
