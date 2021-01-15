from typing import Optional

import torch
import torch.nn as nn

from models.layers.transformer_layers import TransformerEmbedding
from models.layers.tensor_utils import generate_square_subsequent_mask


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self,
                 hidden_units: int,
                 dropout_rate: float):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self,
                inputs: torch.Tensor
                ) -> torch.Tensor:
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class SASRecModel(nn.Module):
    """
    Implementation of the "Self-Attentive Sequential Recommendation" paper.
    see https://doi.org/10.1109%2fICDM.2018.00035 for more details

    see https://github.com/kang205/SASRec for the original Tensorflow implementation
    """

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 transformer_dropout: float,
                 embedding_pooling_type: str,
                 ):
        """
        inits the SASRec model
        :param transformer_hidden_size: the hidden size of the transformer
        :param num_transformer_heads: the number of heads of the transformer
        :param num_transformer_layers: the number of layers of the transformer
        :param item_vocab_size: the item vocab size
        :param max_seq_length: the max sequence length
        :param transformer_dropout: the dropout of the model
        :param embedding_pooling_type: the pooling to use for basket recommendation
        """
        super().__init__()

        self.transformer_dropout = transformer_dropout
        self.max_seq_length = max_seq_length
        self.transformer_hidden_size = transformer_hidden_size
        self.num_transformer_heads = num_transformer_heads
        self.num_transformer_layers = num_transformer_layers
        self.item_vocab_size = item_vocab_size
        self.embedding_mode = embedding_pooling_type

        self.embedding = TransformerEmbedding(item_voc_size=self.item_vocab_size,
                                              max_seq_len=self.max_seq_length,
                                              embedding_size=self.transformer_hidden_size,
                                              dropout=self.transformer_dropout,
                                              embedding_pooling_type=self.embedding_mode)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.transformer_hidden_size, eps=1e-8)

        for _ in range(self.num_transformer_layers):
            attn_layernorm = torch.nn.LayerNorm(self.transformer_hidden_size, eps=1e-8)
            self.attention_layernorms.append(attn_layernorm)

            attn_layer = nn.MultiheadAttention(self.transformer_hidden_size,
                                               self.num_transformer_heads,
                                               self.transformer_dropout)
            self.attention_layers.append(attn_layer)

            fwd_layernorm = torch.nn.LayerNorm(self.transformer_hidden_size, eps=1e-8)
            self.forward_layernorms.append(fwd_layernorm)

            fwd_layer = PointWiseFeedForward(self.transformer_hidden_size, self.transformer_dropout)
            self.forward_layers.append(fwd_layer)

        self.input_sequence_mask = None

    def forward(self,
                input_sequence: torch.Tensor,
                pos_items: torch.Tensor,
                neg_items: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None):
        """
        Forward pass to generate the logits for the positive (next) items and the negative (randomly sampled items,
        that are not in the current sequence) items.
        If no negative items are provided,

        :param input_sequence: the sequence :math:`(N, S)`
        :param pos_items: ids of the positive items (the next items in the sequence) :math:`(N)`
        :param neg_items: random sampled negative items that are not in the session of the user :math:`(N)`
        :param position_ids: the optional position ids if not the position ids are generated :math:`(N, S)`
        :param padding_mask: the optional padding mask if the sequence is padded :math:`(N, S)`
        :return: the logits of the pos_items and the logits of the negative_items, each of shape :math:`(N, S)`
                iff neg_items is provided else the logits for the provided positive items of the same shape

        Where S is the (max) sequence length of the batch and N the batch size.
        """
        # … and H is the hidden size of the transformer/embedding
        device = input_sequence.device
        if self.input_sequence_mask is None or self.input_sequence_mask.size(1) != input_sequence.size()[1]:
            # to mask the next words we generate a triangle mask
            mask = generate_square_subsequent_mask(input_sequence.size()[1]).to(device)
            self.input_sequence_mask = mask

        # embed the input sequence
        input_sequence = self.embedding(input_sequence, position_ids)  # (N, S, H)

        # pipe the embedded sequence to the transformer

        for i in range(len(self.attention_layers)):
            input_sequence = torch.transpose(input_sequence, 0, 1)
            Q = self.attention_layernorms[i](input_sequence)
            mha_outputs, _ = self.attention_layers[i](Q, input_sequence, input_sequence,
                                                      attn_mask=self.input_sequence_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            input_sequence = Q + mha_outputs
            input_sequence = torch.transpose(input_sequence, 0, 1)

            input_sequence = self.forward_layernorms[i](input_sequence)
            input_sequence = self.forward_layers[i](input_sequence)
            input_sequence *= ~padding_mask.unsqueeze(-1)

        transformer_output = self.last_layernorm(input_sequence)

        # when training the model we multiply the seq embedding with the positive and negative items
        if neg_items is not None:
            emb_pos_items = self.embedding.get_item_embedding(pos_items)  # (N, H)
            emb_neg_items = self.embedding.get_item_embedding(neg_items)  # (N, H)

            pos_output = emb_pos_items * transformer_output  # (N, S, H)
            neg_output = emb_neg_items * transformer_output  # (N, S, H)

            pos_output = torch.sum(pos_output, -1)  # (N, S)
            neg_output = torch.sum(neg_output, -1)  # (N, S)

            return pos_output, neg_output

        # inference step (I is the number of positive items to test)
        # embeddings of pos_items
        item_embeddings = self.embedding.get_item_embedding(pos_items, flatten=False)  # (I, N, H)

        # permute embeddings for batch matrix multiplication
        item_embeddings = item_embeddings.permute(0, 2, 1)  # (N, H, I)

        output = torch.matmul(transformer_output, item_embeddings)  # (N, S, I)

        # we use "advanced" indexing to slice the right elements from the sequence.
        batch_size = output.size()[0]
        batch_index = torch.arange(0, batch_size)

        # calculate indices from the padding mask
        seq_index = (~padding_mask).sum(-1) - 1
        scores = output[batch_index, seq_index, :]  # (N, I)
        return scores
