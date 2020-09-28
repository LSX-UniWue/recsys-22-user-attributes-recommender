import torch
import torch.nn as nn
from pyhocon import ConfigTree


class GRUSeqItemRecommenderModel(nn.Module):

    def __init__(self,
                 num_items: int,
                 item_embedding_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float):
        super(GRUSeqItemRecommenderModel, self).__init__()

        self.item_embeddings = nn.Embedding(num_items, embedding_dim=item_embedding_dim)
        self.gru = nn.GRU(item_embedding_dim, hidden_size, dropout=dropout, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, num_items, bias=True)

    def forward(self, session, lengths, batch_idx):
        embedded_session = self.item_embeddings(session)
        packed_embedded_session = nn.utils.rnn.pack_padded_sequence(
            embedded_session,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        _, final_state = self.gru(packed_embedded_session)

        output = self.fcn(final_state)
        return torch.squeeze(output, dim=0)

    @staticmethod
    def from_configuration(config: ConfigTree):
        model_config = config["model"]["gru"]

        num_items = model_config.get_int("num_items")
        item_embedding_dim = model_config.get_int("item_embedding_dim")
        hidden_size = model_config.get_int("hidden_size")
        num_layers = model_config.get_int("num_layers")
        dropout = model_config.get_float("dropout")

        return GRUSeqItemRecommenderModel(num_items, item_embedding_dim, hidden_size, num_layers, dropout)
