import torch
from torch import nn

from models.layers.layers import ItemEmbedding
from utils.hyperparameter_utils import save_hyperparameters


class NNRecModel(nn.Module):

    """
    NNRec implementation from the paper "Next Basket Recommendation with Neural Networks"

    http://ceur-ws.org/Vol-1441/recsys2015_poster15.pdf
    """

    @save_hyperparameters
    def __init__(self,
                 item_vocab_size: int,
                 user_vocab_size: int,
                 item_embedding_size: int,
                 user_embedding_size: int,
                 hidden_size: int,  # l in the paper
                 max_sequence_length: int,  # k, number of last baskets in the paper
                 embedding_pooling_type: str
                 ):
        super().__init__()

        self.item_embedding = ItemEmbedding(item_vocab_size, item_embedding_size,
                                            embedding_pooling_type=embedding_pooling_type)

        # if we have no user the embedding can be ignored
        self.user_embedding_size = user_embedding_size
        if user_embedding_size > 0:
            self.user_embedding = nn.Embedding(user_vocab_size, user_embedding_size)

        # layer 1 in the paper
        self.hidden_layer = nn.Linear(max_sequence_length * item_embedding_size + user_embedding_size, hidden_size)
        self.act1 = nn.Tanh()

        self.projection_layer = nn.Linear(hidden_size, item_vocab_size)  # layer 2 in the paper

    def forward(self,
                input_sequence: torch.Tensor,
                user: torch.Tensor = None
                ) -> torch.Tensor:
        """
        performs a forward path
        :param input_sequence: input sequence :math`(N, S)`
        :param user: the user id :math`(N)`
        :return: the logits for each item :math`(N, I)`
        """

        embedded_items = self.item_embedding(input_sequence)
        batch_size = embedded_items.size()[0]
        embedded_items = embedded_items.view(batch_size, -1)

        if user is not None:
            embedded_user = self.user_embedding(user)
            overall_representation = torch.cat([embedded_user, embedded_items])
        else:
            overall_representation = embedded_items

        first_hidden = self.act1(self.hidden_layer(overall_representation))

        return self.projection_layer(first_hidden)
