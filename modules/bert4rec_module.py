import torch
import pytorch_lightning as pl

from typing import Tuple, List

from torch import nn

from metrics.utils.metric_utils import build_metrics
from tokenization.tokenizer import Tokenizer
from models.bert4rec.bert4rec_model import BERT4RecModel

CROSS_ENTROPY_IGNORE_INDEX = -100


class BERT4RecModule(pl.LightningModule):

    def __init__(self,
                 model: BERT4RecModel,
                 batch_size: int,
                 learning_rate: float,
                 beta_1: float,
                 beta_2: float,
                 tokenizer: Tokenizer,
                 batch_first: bool,
                 ks: List[int]
                 ):
        super().__init__()
        self.model = model

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        self.tokenizer = tokenizer
        self.batch_first = batch_first
        self.metrics = build_metrics(ks)

    def training_step(self, batch, batch_idx):
        input_seq = batch['session']

        # random mask some items
        input_seq, target = _mask_items(input_seq, self.tokenizer, 0.8)
        # calc the padding mask
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=True)

        # call the model
        prediction_scores = self.model(input_seq, padding_mask=padding_mask)
        loss_func = nn.CrossEntropyLoss(ignore_index=CROSS_ENTROPY_IGNORE_INDEX)
        masked_lm_loss = loss_func(prediction_scores.view(-1, self.config.item_voc_size), target.view(-1))

        return {
            'loss': masked_lm_loss
        }


def _mask_items(inputs: torch.Tensor,
                tokenizer: Tokenizer,
                mask_probability: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked items inputs/target for masked modeling. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This itemizer does not have a mask item which is necessary for masked modeling."
        )

    target = inputs.clone()
    # we sample a few items in all sequences for the mask training
    probability_matrix = torch.full(target.shape, mask_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer.pad_token is not None:
        padding_mask = target.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    target[~masked_indices] = CROSS_ENTROPY_IGNORE_INDEX  # We only compute loss on masked items

    # 80% of the time, we replace masked input items with mask item ([MASK])
    indices_replaced = torch.bernoulli(torch.full(target.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input items with random items
    indices_random = torch.bernoulli(torch.full(target.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), target.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # the rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, target


def get_padding_mask(tensor: torch.Tensor,
                     tokenizer: Tokenizer,
                     transposed: bool = True) -> torch.Tensor:
    """
    generates the padding mask based on the tokenizer (by default batch first)
    :param tensor:
    :param itemizer:
    :param transposed:
    :return:
    """
    # the masking should be true where the padding token is set
    padding_mask = tensor.eq(tokenizer.pad_token_id)

    if transposed:
        return padding_mask.transpose(0, 1)

    return padding_mask
