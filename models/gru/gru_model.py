import torch
import torch.nn as nn

from configs.models.gru.gru_config import GRUConfig


class GRUSeqItemRecommenderModel(nn.Module):

    def __init__(self, config: GRUConfig):
        super(GRUSeqItemRecommenderModel, self).__init__()
        self.config = config
        num_items = config.item_voc_size
        item_embedding_dim = config.gru_token_embedding_size
        hidden_size = config.gru_hidden_size
        num_layers = config.gru_num_layers

        self.item_embeddings = nn.Embedding(num_items, embedding_dim=item_embedding_dim)
        self.gru = nn.GRU(item_embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
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
