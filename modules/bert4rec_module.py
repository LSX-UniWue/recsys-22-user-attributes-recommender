import torch
import pytorch_lightning as pl

from typing import Tuple

from torch import nn

from configs.models.bert4rec.bert4rec_config import BERT4RecConfig
from configs.training.bert4rec.bert4rec_config import BERT4RecTrainingConfig
from tokenization.tokenizer import Tokenizer
from models.bert4rec.bert4rec_model import BERT4RecModel
from module_registry import module_registry

CROSS_ENTROPY_IGNORE_INDEX = -100


@module_registry.register_module('bert4rec')
class BERT4RecModule(pl.LightningModule):

    def __init__(self,
                 training_config: BERT4RecTrainingConfig,
                 model_config: BERT4RecConfig
                 ):
        super().__init__()

        self.training_config = training_config
        self.model_config = model_config
        self.model = BERT4RecModel(model_config)

    # TODO: discuss: here the bert4rec model needs the itemizer
    def training_step(self, batch, batch_idx, itemizer):
        input_seq = batch['sequence']

        # random mask some items
        input_seq, target = _mask_items(input_seq, itemizer, 0.8)
        # calc the padding mask
        padding_mask = get_padding_mask(input_seq, itemizer, transposed=True)

        # call the model
        prediction_scores = self.model(input_seq, padding_mask=padding_mask)
        loss_func = nn.CrossEntropyLoss(ignore_index=CROSS_ENTROPY_IGNORE_INDEX)
        masked_lm_loss = loss_func(prediction_scores.view(-1, self.config.item_voc_size), target.view(-1))

        return {
            'loss': masked_lm_loss
        }


def _mask_items(inputs: torch.Tensor,
                itemizer: Tokenizer,
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
    target[~masked_indices] = CROSS_ENTROPY_IGNORE_INDEX  # We only compute loss on masked items

    # 80% of the time, we replace masked input items with mask item ([MASK])
    indices_replaced = torch.bernoulli(torch.full(target.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = itemizer.convert_tokens_to_ids(itemizer.mask_token)

    # 10% of the time, we replace masked input items with random items
    indices_random = torch.bernoulli(torch.full(target.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(itemizer), target.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # the rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, target


def get_padding_mask(tensor: torch.Tensor,
                     itemizer: Tokenizer,
                     transposed: bool = True) -> torch.Tensor:
    """
    generates the padding mask based on the itemizer (by default batch first)
    :param tensor:
    :param itemizer:
    :param transposed:
    :return:
    """
    # the masking should be true where the paddding token is set
    padding_mask = tensor.eq(itemizer.pad_token_id)

    if transposed:
        return padding_mask.transpose(0, 1)

    return padding_mask
