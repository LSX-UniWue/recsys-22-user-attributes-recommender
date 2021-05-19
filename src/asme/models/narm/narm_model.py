from typing import Optional, List

import torch
import torch.nn as nn

from asme.models.layers.layers import ItemEmbedding
from asme.models.sequence_recommendation_model import SequenceRecommenderModel
from asme.utils.hyperparameter_utils import save_hyperparameters


class NarmModel(SequenceRecommenderModel):
    """
        Implementation of "Neural Attentive Session-based Recommendation." (https://dl.acm.org/doi/10.1145/3132847.3132926).

        See https://github.com/lijingsdu/sessionRec_NARM for the original Theano implementation.

        Shapes:
        * N - batch size
        * I - number of items
        * E - item embedding size
        * H - representation size of the encoder
        * S - sequence length
    """

    @save_hyperparameters
    def __init__(self,
                 item_vocab_size: int,
                 item_embedding_size: int,
                 global_encoder_size: int,
                 global_encoder_num_layers: int,
                 embedding_dropout: float,
                 context_dropout: float,
                 batch_first: bool = True,
                 embedding_pooling_type: str = None):

        """
        :param item_vocab_size: number of items (I)
        :param item_embedding_size: item embedding size (E)
        :param global_encoder_size: hidden size of the GRU used as the encoder (H)
        :param global_encoder_num_layers: number of layers of the encoder GRU
        :param embedding_dropout: dropout applied after embedding the items
        :param context_dropout: dropout applied on the full context representation
        :param batch_first: whether data is batch first.
        :param embedding_pooling_type: the embedding mode to use if multiple items per
        """
        super(NarmModel, self).__init__()
        self.batch_first = batch_first
        self.item_embeddings = ItemEmbedding(item_voc_size=item_vocab_size,
                                             embedding_size=item_embedding_size,
                                             embedding_pooling_type=embedding_pooling_type)
        self.item_embedding_dropout = nn.Dropout(embedding_dropout)
        self.global_encoder = nn.GRU(item_embedding_size, global_encoder_size,
                                     num_layers=global_encoder_num_layers,
                                     batch_first=batch_first)
        self.local_encoder = LocalEncoderLayer(global_encoder_size, global_encoder_size)
        self.context_dropout = nn.Dropout(context_dropout)
        self.decoder = BilinearDecoder(self.item_embeddings, encoded_representation_size=2 * global_encoder_size)

    def forward(self,
                sequence: torch.Tensor,
                padding_mask: Optional[torch.Tensor],
                **kwargs
                ):
        """
        Computes item similarity scores using the NARM model.

        :param sequence: a sequence of item ids :math`(N, S)`
        :param padding_mask: a tensor masking padded sequence elements :math`(N, S)`
        :return: scores for every item :math`(N, I)`

        where N is the batch size, S the max sequence length, I the item vocab size
        """
        # H is the hidden size of the model
        embedded_sequence = self.item_embeddings(sequence)  # (N, S, H)
        embedded_sequence = self.item_embedding_dropout(embedded_sequence)  # (N, S, H)

        max_seq_length = sequence.size()[1]  # should be S
        lengths = padding_mask.sum(dim=-1).cpu()
        packed_embedded_session = nn.utils.rnn.pack_padded_sequence(
            embedded_sequence,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        h_i, h_t = self.global_encoder(packed_embedded_session)
        c_tg = h_t = h_t[-1]  # we use the last hidden state of the last layer

        # we only use the hidden size and throw away the lengths, since we already have them
        h_i, _ = nn.utils.rnn.pad_packed_sequence(h_i, batch_first=self.batch_first, total_length=max_seq_length)
        c_tl = self.local_encoder(h_t, h_i, padding_mask)

        c_t = torch.cat([c_tg, c_tl], dim=1)

        c_tdo = self.context_dropout(c_t)

        return self.decoder(c_tdo)


class LocalEncoderLayer(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 latent_size: int
                 ):
        """
        Local encoder as defined in "Neural Attentive Session-based Recommendation." (https://dl.acm.org/doi/10.1145/3132847.3132926).

        :param hidden_size: hidden size of the encoder RNN.
        :param latent_size: size of the latent space used for computing the alphas.
        """
        super(LocalEncoderLayer, self).__init__()
        self.A1 = nn.Linear(hidden_size, latent_size, bias=False)
        self.A2 = nn.Linear(hidden_size, latent_size, bias=False)
        self.v = nn.Parameter(torch.Tensor(latent_size))

        self.projection_activation = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.uniform_(self.v, -1.0, 1.0)

    def forward(self,
                s1: torch.Tensor,
                s2: torch.Tensor,
                mask: torch.Tensor
                ):
        """
        Calculates v * sigmoid(A1 * s1 + A2 * s2)
        :param s1: a tensor (N, H)
        :param s2: a tensor (N, S, H)
        :param mask: a tensor TODO: alex: please add the dimension

        :return: :math `(N, S, 1)`
        """
        s1_prj = self.A1(s1)  # (N, H)
        s2_prj = self.A2(s2)  # (N, S, H)

        # we need to repeat s_1 for every step in the batch to calculate all steps at once
        s1_prj = torch.unsqueeze(s1_prj, dim=1)  # (N, 1, H)
        s1_prj = torch.repeat_interleave(s1_prj, repeats=s2_prj.size()[1], dim=1)  # (N, S, H)
        sum_projection = s1_prj + s2_prj
        state_representation = self.projection_activation(sum_projection)
        alphas = torch.matmul(state_representation, self.v)  # (N, S),  (v: (1, H) * (N, S, H))
        alphas = torch.unsqueeze(alphas, dim=2)  # (N, S, 1) one scalar weight for every position in the sequence

        weighted = alphas * s2
        # (N, S, H): the alphas get broadcasted along the last dimension and then a component-wise multiplication
        # is performed, scaling every step differently

        # (N, S, 1) make sure that the mask value for each sequence member is broadcasted to all features
        # -> zeroing out masked features
        mask = torch.unsqueeze(mask, dim=-1)

        weighted = mask.to(dtype=weighted.dtype) * weighted  # (N, S, H)

        return torch.sum(weighted, dim=1)  # (N, H)


class BilinearDecoder(nn.Module):
    """
        Implementation of the bilinear decoder
    """
    def __init__(self,
                 embedding_layer: ItemEmbedding,
                 encoded_representation_size: int,
                 apply_softmax: bool = False
                 ):
        """

        :param embedding_layer: an item embedding layer
        :param encoded_representation_size: the full size of the encoded representation
        :param apply_softmax: whether to apply softmax or not.
        """
        super().__init__()

        self.embedding_layer = embedding_layer
        # TODO: rename
        self.B = nn.Linear(embedding_layer.embedding.weight.size()[1], encoded_representation_size, bias=False)
        self.activation = nn.Softmax() if apply_softmax else nn.Identity()

    def forward(self,
                context: torch.Tensor,
                items: torch.Tensor = None
                ):
        """
        Computes the similarity scores for the context with each item in 'items'.
        If 'items' is none a similarity score for every item will be computed.

        :param context: a context tensor (N, H).
        :param items: a tensor with items (NI).

        :return: the similarity scores (N, NI),

        where N is the batch size and NI the number of items (can be I)
        """
        # context: (N, H)
        # self.B: (E, H)

        if not items:
            items = torch.arange(self.embedding_layer.embedding.weight.size()[0], dtype=torch.long, device=context.device)
        # items: NI <- number of items evaluated

        embi = self.embedding_layer(items, flatten=False)  # (NI, E)
        embi_b = self.B(embi)  # (NI, H)
        embi_b_T = embi_b.T  # (H, NI)

        similarities = torch.matmul(context, embi_b_T)  # (N, NI) <- a score for each item

        return self.activation(similarities)
