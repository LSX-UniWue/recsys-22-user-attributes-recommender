from typing import Optional, Union, Tuple

import torch
import torch.nn as nn

from models.layers.transformer_layers import TransformerEmbedding, TransformerLayer


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

        self.transformer_encoder = TransformerLayer(transformer_hidden_size, num_transformer_heads,
                                                    num_transformer_layers, transformer_hidden_size * 4,
                                                    transformer_dropout)

    def forward(self,
                sequence: torch.Tensor,
                pos_items: torch.Tensor,
                neg_items: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None
                ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass to generate the logits for the positive (next) items and the negative (randomly sampled items,
        that are not in the current sequence) items.
        If no negative items are provided,

        :param sequence: the sequence :math:`(N, S)`
        :param pos_items: ids of the positive items (the next items in the sequence) :math:`(N)`
        :param neg_items: random sampled negative items that are not in the session of the user :math:`(N)`
        :param position_ids: the optional position ids if not the position ids are generated :math:`(N, S)`
        :param padding_mask: the optional padding mask if the sequence is padded :math:`(N, S)` True if not padded
        :return: the logits of the pos_items and the logits of the negative_items, each of shape :math:`(N, S)`
                iff neg_items is provided else the logits for the provided positive items of the same shape

        Where S is the (max) sequence length of the batch and N the batch size.
        """

        # embed the input sequence
        embedded_sequence = self.embedding(sequence, position_ids)  # (N, S, H)

        # pipe the embedded sequence to the transformer
        # first build the attention mask FIXME: check the attention mask
        input_size = sequence.size()
        batch_size = input_size[0]
        sequence_length = input_size[1]

        attention_mask = torch.triu(torch.ones([sequence_length, sequence_length], device=sequence.device))\
            .transpose(1, 0).unsqueeze(0).repeat(batch_size, 1, 1)
        if padding_mask is not None:
            attention_mask = attention_mask * padding_mask.unsqueeze(1).repeat(1, sequence_length, 1)
        attention_mask = attention_mask.unsqueeze(1).to(dtype=torch.bool)

        transformer_output = self.transformer_encoder(embedded_sequence, attention_mask=attention_mask)

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
        item_embeddings = self.embedding.get_item_embedding(pos_items, flatten=False)  # (N, I, H)

        # we use "advanced" indexing to slice the right elements from the transformer output
        batch_size = sequence.size()[0]
        batch_index = torch.arange(0, batch_size)

        # calculate indices from the padding mask
        seq_index = padding_mask.sum(-1) - 1
        transformer_last_pos_output = transformer_output[batch_index, seq_index]  # (N, H)

        # now matmul it with the item embeddings
        logits = item_embeddings.matmul(transformer_last_pos_output.unsqueeze(-1))

        return logits.squeeze(-1)
