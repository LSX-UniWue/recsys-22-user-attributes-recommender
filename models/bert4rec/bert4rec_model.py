from typing import Tuple

import torch
from torch import nn

from config.bert4rec.bert4rec_config import BERT4RecConfig
from models.layers.util_layers import TransformerEmbedding, MatrixFactorizationLayer
from utils.itemization_utils import PreTrainedItemizer

CROSS_ENTROPY_IGNORE_INDEX = -100


class BERT4RecModel(nn.Module):
    """
    implementation of the paper "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"
    see https://doi.org/10.1145%2f3357384.3357895 for more details.
    """

    def __init__(self, config: BERT4RecConfig):
        super().__init__()

        self.d_model = config.d_model
        self.embedding = TransformerEmbedding(config.item_voc_size, config.max_seq_length, self.d_model)

        encoder_layers = nn.TransformerEncoderLayer(self.d_model, config.num_transformer_heads, self.d_model,
                                                    config.transformer_dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.num_transformer_layers)

        # for decoding the sequence into the item space again
        self.linear = nn.Linear(self.d_model, self.d_model)
        self.gelu = nn.GELU()
        self.mfl = MatrixFactorizationLayer(self.embedding.get_item_embedding_weight())
        self.softmax = nn.Softmax(dim=2)

    def forward(self,
                input_seq: torch.Tensor,
                position_ids: torch.Tensor = None,
                padding_mask: torch.Tensor = None):
        """
        forward pass to calculate the scores for the mask item modelling

        :param input_seq: the input sequence [S x B]
        :param position_ids: (optional) positional_ids if None the position ids are generated [S x B]
        :param padding_mask: (optional) the padding mask if the sequence is padded [B x S]
        :return: the scores of the predicted tokens [S x B x I] (Note: here all scores for all positions are returned.
        For loss calculation please only use MASK tokens.)

        Where S is the (max) sequence length of the batch, B the batch size, and I the vocabulary size of the items.
        """
        # embed the input
        input_seq = self.embedding(input_seq, position_ids)

        # use the bidirectional transformer
        input_seq = self.transformer_encoder(input_seq,
                                             src_key_padding_mask=padding_mask)

        # decode the hidden representation
        scores = self.gelu(self.linear(input_seq))

        scores = self.mfl(scores)
        scores = self.softmax(scores)
        return scores


def _mask_items(inputs: torch.Tensor,
               itemizer: PreTrainedItemizer,
               mask_probability: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked items inputs/target for masked modeling. """

    if itemizer.mask_token is None:
        raise ValueError(
            "This itemizer does not have a mask item which is necessary for masked modeling."
        )

    target = inputs.clone()
    # we sample a few items in all sequences for the mask training
    probability_matrix = torch.full(target.shape, mask_probability)
    special_tokens_mask = [
        itemizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if itemizer.pad_token is not None:
        padding_mask = target.eq(itemizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    target[~masked_indices] = CROSS_ENTROPY_IGNORE_INDEX  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input items with mask item ([MASK])
    indices_replaced = torch.bernoulli(torch.full(target.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = itemizer.convert_items_to_ids(itemizer.mask_token)

    # 10% of the time, we replace masked input items with random items
    indices_random = torch.bernoulli(torch.full(target.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(itemizer), target.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # the rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, target


def _get_padding_mask(tensor: torch.Tensor,
                      itemizer: PreTrainedItemizer) -> torch.Tensor:
    # the masking should be true where the paddding token is set
    padding_mask = tensor.eq(itemizer.pad_token_id)
    # here the transformer required a batch first tensor, so we transform it here
    return padding_mask.transpose(0, 1)
