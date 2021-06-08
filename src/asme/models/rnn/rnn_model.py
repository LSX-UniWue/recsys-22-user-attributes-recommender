from abc import abstractmethod
from typing import Tuple, Optional, List, Union

import torch
from torch import nn

from asme.models.components.sequence_embedding import SequenceElementsEmbeddingComponent
from asme.models.layers.data.sequence import EmbeddedElementsSequence, SequenceRepresentation, \
    ModifiedSequenceRepresentation
from asme.models.layers.layers import PROJECT_TYPE_LINEAR, build_projection_layer, SequenceRepresentationLayer, \
    ProjectionLayer, IdentitySequenceRepresentationModifierLayer
from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.utils.hyperparameter_utils import save_hyperparameters


def _build_rnn_cell(cell_type: str,
                    item_embedding_size: int,
                    hidden_size: int,
                    num_layers: int,
                    bidirectional: bool,
                    dropout: float,
                    nonlinearity: str  # only for Elman RNN
                    ) -> nn.Module:
    if cell_type == 'gru':
        return nn.GRU(item_embedding_size, hidden_size,
                      bidirectional=bidirectional,
                      dropout=dropout,
                      num_layers=num_layers,
                      batch_first=True)

    if cell_type == 'lstm':
        return LSTMAdapter(item_embedding_size, hidden_size,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           num_layers=num_layers)

    if cell_type == 'rnn':
        return nn.RNN(item_embedding_size, hidden_size,
                      bidirectional=bidirectional,
                      dropout=dropout,
                      num_layers=num_layers,
                      nonlinearity=nonlinearity)

    raise ValueError(f'cell type "{cell_type}" not supported')


class LSTMAdapter(nn.Module):
    """
    Changes the output of `torch.nn.LSTM` to comply with the API for `torch.nn.GRU` by omitting the internal
    context states `c_n`.
    """
    def __init__(self,
                 item_embedding_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bidirectional: bool,
                 dropout: float
                 ):
        super().__init__()

        self.lstm = nn.LSTM(item_embedding_size, hidden_size,
                            bidirectional=bidirectional,
                            dropout=dropout,
                            num_layers=num_layers,
                            batch_first=True)

    def forward(self,
                packed_embedded_session: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, final_state = self.lstm(packed_embedded_session)

        # currently we ignore the context information and only return
        # the hidden representation (as the gru model)
        return outputs, final_state[0]


class RNNSequenceRepresentationComponent(SequenceRepresentationLayer):

    def __init__(self,
                 cell_type: str,
                 item_embedding_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 bidirectional: bool = False,
                 nonlinearity: str = None  # for Elman RNN
                 ):

        super().__init__()

        # FIXME: maybe this should not be done here
        rnn_dropout = dropout
        if num_layers == 1 and dropout > 0:
            print("setting the dropout of the rnn to 0 because the number of layers is 1")
            rnn_dropout = 0.0

        self.rnn = _build_rnn_cell(cell_type, item_embedding_dim, hidden_size, num_layers, bidirectional, rnn_dropout,
                                   nonlinearity)

        self.pooling = RNNPooler(bidirectional=bidirectional)

    def forward(self, embedded_sequence: EmbeddedElementsSequence) -> SequenceRepresentation:
        input_sequence = embedded_sequence.input_sequence

        lengths = input_sequence.padding_mask.sum(dim=-1).cpu()  # required by torch >= 1.7, no cuda tensor allowed
        packed_embedded_session = nn.utils.rnn.pack_padded_sequence(
            embedded_sequence,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        outputs, final_state = self.rnn(packed_embedded_session)
        output = self.pooling(outputs, final_state)

        return SequenceRepresentation(output, embedded_sequence)


class RNNProjectionComponent(ProjectionLayer):

    def __init__(self,
                 item_vocab_size: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 project_layer_type: str = PROJECT_TYPE_LINEAR):
        super().__init__()
        hidden_size_projection = hidden_size if not bidirectional else 2 * hidden_size
        self.projection = build_projection_layer(project_layer_type, hidden_size_projection, item_vocab_size,
                                                 self.item_embedding.embedding)

    def forward(self, modified_sequence_representation: ModifiedSequenceRepresentation) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:
        sequence_representation = modified_sequence_representation.modified_encoded_sequence
        return self.projection(sequence_representation)


class RNNModel(SequenceRecommenderModel):

    @save_hyperparameters
    def __init__(self,
                 cell_type: str,
                 item_vocab_size: int,
                 item_embedding_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 bidirectional: bool = False,
                 nonlinearity: str = None,  # for Elman RNN
                 embedding_pooling_type: str = None,
                 project_layer_type: str = PROJECT_TYPE_LINEAR):

        sequence_embedding_component = SequenceElementsEmbeddingComponent(vocabulary_size=item_vocab_size,
                                                                          embedding_size=item_embedding_dim,
                                                                          pooling_type=embedding_pooling_type)

        sequence_representation_component = RNNSequenceRepresentationComponent(cell_type,
                                                                               item_embedding_dim,
                                                                               hidden_size,
                                                                               num_layers,
                                                                               dropout,
                                                                               bidirectional,
                                                                               nonlinearity)

        projection_component = RNNProjectionComponent(item_vocab_size, hidden_size, bidirectional, project_layer_type)

        super().__init__(sequence_embedding_component,
                         sequence_representation_component,
                         IdentitySequenceRepresentationModifierLayer(),
                         projection_component)


class RNNStatePooler(nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self,
                outputs: torch.Tensor,
                hidden_representation: torch.Tensor
                ) -> torch.Tensor:
        pass


class RNNPooler(RNNStatePooler):

    def __init__(self,
                 bidirectional: bool = False
                 ):
        super().__init__()

        self.directions = 2 if bidirectional else 1

    def forward(self,
                outputs: torch.Tensor,
                hidden_representation: torch.Tensor
                ) -> torch.Tensor:
        if self.directions == 1:
            # we "pool" the model by simply taking the hidden state of the last layer
            # of an unidirectional model
            return hidden_representation[-1, :, :]

        # FIXME: discuss this representation (state of last layer, both directions)
        layer_direction, batch_size, hidden_size = hidden_representation.size()
        layers = int(layer_direction / self.directions)

        hidden_representation = hidden_representation.view(layers, self.directions, batch_size, hidden_size)

        last_layer_representation = hidden_representation[-1]
        representation = torch.cat([last_layer_representation[0], last_layer_representation[1]], dim=1)
        return representation
