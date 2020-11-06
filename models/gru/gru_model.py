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
                 embedding_mode: str = None
                 ):
        super(GRUSeqItemRecommenderModel, self).__init__()

        self.embedding_mode = embedding_mode

        self.item_embeddings = ItemEmbedding(item_voc_size=num_items,
                                             embedding_size=item_embedding_dim,
                                             embedding_mode=self.embedding_mode)

        # FIXME: maybe this should not be done here
        if num_layers == 1 and dropout > 0:
            print("setting the dropout to 0 because the number of layers is 1")
            dropout = 0

        self.gru = nn.GRU(item_embedding_dim, hidden_size, dropout=dropout, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, num_items, bias=True)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self,
                session: torch.Tensor,
                mask: torch.Tensor
                ) -> torch.Tensor:
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
