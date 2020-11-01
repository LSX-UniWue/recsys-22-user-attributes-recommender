import torch
import torch.nn as nn

from models.layers.layers import ItemEmbedding


class GRUSeqItemRecommenderModel(nn.Module):

    def __init__(self,
                 num_items: int,
                 item_embedding_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 embedding_mode: str = None):
        super(GRUSeqItemRecommenderModel, self).__init__()

        self.embedding_mode = embedding_mode

        self.item_embeddings = ItemEmbedding(item_voc_size=self.item_voc_size,
                                             embedding_size=self.embedding_size,
                                             embedding_mode=self.embedding_mode)
        self.gru = nn.GRU(item_embedding_dim, hidden_size, dropout=dropout, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, num_items, bias=True)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, session, mask: torch.Tensor, batch_idx):
        embedded_session = self.item_embeddings(session)
        embedded_session = self.dropout(embedded_session)
        packed_embedded_session = nn.utils.rnn.pack_padded_sequence(
            embedded_session,
            torch.sum(mask, dim=-1),
            batch_first=True,
            enforce_sorted=False
        )
        _, final_state = self.gru(packed_embedded_session)

        output = self.fcn(final_state)
        return torch.squeeze(output, dim=0)
