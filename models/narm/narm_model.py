import torch
import torch.nn as nn

from models.layers.layers import ItemEmbedding


class NarmModel(nn.Module):
    """
        Implementation of "Neural Attentive Session-based Recommendation." (https://dl.acm.org/doi/10.1145/3132847.3132926).

        See https://github.com/lijingsdu/sessionRec_NARM for the original Theano implementation.

        Shapes:
        * B - batch size
        * NI - number of items
        * E - item embedding size
        * H - representation size of the encoder
        * N - sequence length
    """

    def __init__(self,
                 num_items: int,
                 item_embedding_size: int,
                 global_encoder_size: int,
                 global_encoder_num_layers: int,
                 embedding_dropout: float,
                 context_dropout: float,
                 batch_first: bool = True,
                 embedding_mode: str = None):

        """
        :param num_items: number of items (NI)
        :param item_embedding_size: item embedding size (E)
        :param global_encoder_size: hidden size of the GRU used as the encoder (H)
        :param global_encoder_num_layers: number of layers of the encoder GRU
        :param embedding_dropout: dropout applied after embedding the items
        :param context_dropout: dropout applied on the full context representation
        :param batch_first: whether data is batch first.
        :param embedding_mode: the embedding mode to use if multiple items per
        """
        super(NarmModel, self).__init__()
        self.batch_first = batch_first
        self.item_embeddings = ItemEmbedding(item_voc_size=num_items,
                                             embedding_size=item_embedding_size,
                                             embedding_mode=embedding_mode)
        self.item_embedding_dropout = nn.Dropout(embedding_dropout)
        self.global_encoder = nn.GRU(item_embedding_size, global_encoder_size,
                                     num_layers=global_encoder_num_layers,
                                     batch_first=batch_first)
        self.local_encoder = LocalEncoderLayer(global_encoder_size, global_encoder_size)
        self.context_dropout = nn.Dropout(context_dropout)
        self.decoder = BilinearDecoder(self.item_embeddings, encoded_representation_size=2 * global_encoder_size)

    def forward(self,
                session: torch.Tensor,
                mask: torch.Tensor
                ):
        """
        Computes item similarity scores using the NARM model.

        :param session: a sequence of item ids (B x N)
        :param mask: a tensor masking padded sequence elements (B x N)
        :return: scores for every item (B x NI)
        """
        # H is the hidden size of the model
        embedded_session = self.item_embeddings(session)  # (N, S, H)
        embedded_session_do = self.item_embedding_dropout(embedded_session)  # (N, S, H)

        max_seq_length = embedded_session_do.size()[1]
        packed_embedded_session = nn.utils.rnn.pack_padded_sequence(
            embedded_session_do,
            torch.sum(mask, dim=-1),
            batch_first=True,
            enforce_sorted=False
        )

        h_i, h_t = self.global_encoder(packed_embedded_session)
        c_tg = h_t = torch.squeeze(h_t, dim=0)

        h_i, _ = nn.utils.rnn.pad_packed_sequence(h_i, batch_first=self.batch_first, total_length=max_seq_length)  # we throw away the lengths, since we already have them.
        c_tl = self.local_encoder(h_t, h_i, mask)

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
        self.v = torch.nn.Parameter(torch.Tensor(latent_size))

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
        :param s1: a tensor (B x H)
        :param s2: a tensor (B x N x H)
        :param mask: a tensor TODO: alex: please add the dimension

        :return: (B x N x 1)
        """
        s1_prj = self.A1(s1)  # B x H
        s2_prj = self.A2(s2)  # B x N x H

        # we need to repeat s_1 for every step in the batch to calculate all steps at once
        s1_prj = torch.unsqueeze(s1_prj, dim=1)  # B x 1 x H
        s1_prj = torch.repeat_interleave(s1_prj, repeats=s2_prj.size()[1], dim=1)  # B x N x H
        alphas = torch.matmul(torch.sigmoid(s1_prj + s2_prj), self.v)  # B x N (v: 1 x H * B x N x H)
        alphas = torch.unsqueeze(alphas, dim=2)  # B x N x 1 (one scalar weight for every position in the sequence)

        weighted = alphas * s2
        # B x N x H: the alphas get broadcasted along the last dimension and then a component-wise multiplication
        # is performed, scaling every step differently

        # B x N x 1 make sure that the mask value for each sequence member is broadcasted to all features
        # -> zeroing out masked features
        mask = torch.unsqueeze(mask, dim=-1)

        weighted = mask.to(dtype=weighted.dtype) * weighted  # B x N x H

        return torch.sum(weighted, dim=1)  # B x H


class BilinearDecoder(nn.Module):
    """
        Implementation of the bilinear decoder from "Neural Attentive Session-based Recommendation."
        (https://dl.acm.org/doi/10.1145/3132847.3132926).

        See https://github.com/lijingsdu/sessionRec_NARM for the original Theano implementation.
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
        super(BilinearDecoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.B = nn.Linear(embedding_layer.embedding.weight.size()[1], encoded_representation_size, bias=False)
        self.activation = nn.Softmax() if apply_softmax else nn.Identity()

    def forward(self,
                context: torch.Tensor,
                items: torch.Tensor = None
                ):
        """
        Computes the similarity scores for the context with each item in 'items'.
        If 'items' is none a similarity score for every item will be computed.

        :param context: a context tensor (B x H).
        :param items: a tensor with items (NI).

        :return: the similarity scores (B x NI)
        """
        # context: B x H
        # B: E x H

        if not items:
            items = torch.arange(self.embedding_layer.embedding.weight.size()[0], dtype=torch.long, device=context.device)
        # items: NI <- number of items evaluated

        embi = self.embedding_layer(items, flatten=False)  # NI x E
        embi_b = self.B(embi)  # NI x H
        embi_b_T = embi_b.T  # H x NI

        similarities = torch.matmul(context, embi_b_T)  # B x NI <- a score for each item

        return self.activation(similarities)
