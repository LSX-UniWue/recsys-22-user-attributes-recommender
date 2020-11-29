from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn

from models.layers.layers import ItemEmbedding


def _build_rnn_cell(cell_type: str,
                    item_embedding_size: int,
                    hidden_size: int,
                    num_layers: int,
                    dropout: float
                    ) -> nn.Module:
    if cell_type == 'gru':
        return nn.GRU(item_embedding_size, hidden_size, dropout=dropout, num_layers=num_layers, batch_first=True)

    if cell_type == 'lstm':
        return LSTMSeqItemRecommenderModule(item_embedding_size, hidden_size, dropout=dropout, num_layers=num_layers)

    raise ValueError(f'cell type "{cell_type}" not supported')


class LSTMSeqItemRecommenderModule(nn.Module):

    def __init__(self,
                 item_embedding_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float
                 ):
        super().__init__()

        self.lstm = nn.LSTM(item_embedding_size, hidden_size, dropout=dropout, num_layers=num_layers, batch_first=True)

    def forward(self,
                packed_embedded_session: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, final_state = self.lstm(packed_embedded_session)

        # currently we ignore the context information and only return
        # the hidden representation (as the gru model)
        return outputs, final_state[0]


class RNNSeqItemRecommenderModel(nn.Module):

    def __init__(self,
                 cell_type: str,
                 item_vocab_size: int,
                 item_embedding_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 embedding_mode: str = None):
        super().__init__()
        self.embedding_mode = embedding_mode

        self.item_embeddings = ItemEmbedding(item_voc_size=item_vocab_size,
                                             embedding_size=item_embedding_dim,
                                             embedding_mode=self.embedding_mode)

        # FIXME: maybe this should not be done here
        if num_layers == 1 and dropout > 0:
            print("setting the dropout to 0 because the number of layers is 1")
            dropout = 0

        self.rnn = _build_rnn_cell(cell_type, item_embedding_dim, hidden_size, num_layers, dropout)
        self.pooling = RNNPooler(hidden_size, item_vocab_size)

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
        outputs, final_state = self.rnn(packed_embedded_session)
        output = self.pooling(outputs, final_state)
        return output


class RNNStatePooler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,
                outputs: torch.Tensor,
                hidden_representation: torch.Tensor
                ) -> torch.Tensor:
        return self._pool(outputs, hidden_representation)

    @abstractmethod
    def _pool(self,
              outputs: torch.Tensor,
              hidden_representation: torch.Tensor
              ) -> torch.Tensor:
        pass


class RNNPooler(RNNStatePooler):

    def __init__(self,
                 hidden_size: int,
                 num_items: int
                 ):
        super().__init__()

        self.fcn = nn.Linear(hidden_size, num_items, bias=True)

    def _pool(self, outputs: torch.Tensor, hidden_representation: torch.Tensor) -> torch.Tensor:
        # we "pool" the model by simply taking the hidden state of the last layer
        representation = hidden_representation[-1, :, :]
        return self.fcn(representation)
