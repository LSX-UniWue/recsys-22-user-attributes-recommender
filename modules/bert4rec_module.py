import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from typing import Tuple, List, Union, Dict

from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from tokenization.tokenizer import Tokenizer
from models.bert4rec.bert4rec_model import BERT4RecModel

CROSS_ENTROPY_IGNORE_INDEX = -100


class BERT4RecModule(pl.LightningModule):

    def __init__(self,
                 model: BERT4RecModel,
                 mask_probability: float,
                 learning_rate: float,
                 beta_1: float,
                 beta_2: float,
                 weight_decay: float,
                 num_warmup_steps: int,
                 tokenizer: Tokenizer,
                 batch_first: bool,
                 metrics: torch.nn.ModuleDict
                 ):
        super().__init__()
        self.model = model

        self.mask_probability = mask_probability

        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps

        self.tokenizer = tokenizer
        self.batch_first = batch_first
        self.metrics = metrics

    def training_step(self, batch, batch_idx):
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        input_seq = _expand_sequence(inputs=input_seq, tokenizer=self.tokenizer, batch_first=self.batch_first)

        if self.batch_first:
            input_seq = input_seq.transpose(0, 1)

        # random mask some items
        # FIXME: paper quote: we also produce samples that only mask the last item
        # in the input sequences during training.
        # how? TODO: check code!
        input_seq, target = _mask_items(inputs=input_seq,
                                        tokenizer=self.tokenizer,
                                        mask_probability=self.mask_probability)
        # calc the padding mask
        padding_mask = get_padding_mask(tensor=input_seq,
                                        tokenizer=self.tokenizer,
                                        transposed=True)

        # call the model
        prediction_scores = self.model(input_seq, padding_mask=padding_mask)

        if len(input_seq.size()) > 2:
            pos_weight = torch.ones(len(self.tokenizer), dtype=torch.float, device=prediction_scores.device)
            pos_weight[self.tokenizer.get_special_token_ids()] = 0.0
            loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            # replace the -100 masking from the normal sequence token with the pad token
            # (which is set to be ignored, see pos_weights; padding token is a special token)
            target[target == CROSS_ENTROPY_IGNORE_INDEX] = self.tokenizer.pad_token_id
            # than convert the targets to a multi one hot encoding
            targets = torch.zeros(prediction_scores.size()).scatter_(2, target, 1.)
            targets = targets.squeeze(0)
            targets = targets.float()
            masked_lm_loss = loss_func(prediction_scores, targets)
        else:
            loss_func = nn.CrossEntropyLoss(ignore_index=CROSS_ENTROPY_IGNORE_INDEX)
            flatten_predictions = prediction_scores.view(-1, len(self.tokenizer))
            flatten_targets = torch.flatten(target)
            masked_lm_loss = loss_func(flatten_predictions, flatten_targets)

        return {
            'loss': masked_lm_loss
        }

    def validation_step(self, batch, batch_idx):
        # shorter to allow the masking token
        input_seq = _expand_sequence(inputs=batch[ITEM_SEQ_ENTRY_NAME],
                                     tokenizer=self.tokenizer,
                                     batch_first=self.batch_first)
        targets = batch[TARGET_ENTRY_NAME]

        # set the last non padding token to the mask token
        input_seq, target_mask = _add_mask_token_at_ending(input_seq, self.tokenizer)

        if self.batch_first:
            input_seq = input_seq.transpose(1, 0)
            target_mask = target_mask.transpose(1, 0)

        # after adding the mask token we can calculate the padding mask
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=True)

        # get predictions for all seq steps
        prediction = self.model(input_seq, padding_mask=padding_mask)
        # extract the relevant seq steps, where the mask was set
        prediction = prediction[target_mask]

        for name, metric in self.metrics.items():
            mask = None
            # when we have mulitple target per sequence step, we have to provide a mask for the paddings applied to
            # the target tensor
            if len(input_seq.size()) > 2:
                mask = ~ targets.eq(self.tokenizer.pad_token_id)
            step_value = metric(prediction, targets, mask=mask)
            self.log(name, step_value, prog_bar=True)

    # FIXME: copy paste code from sas rec module
    def validation_epoch_end(self, outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]) -> None:
        for name, metric in self.metrics.items():
            self.log(name, metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        def _filter(name: str) -> bool:
            return name.endswith("bias") or 'norm1' in name or 'norm2' in name or 'layer_norm' in name

        decay_exclude = [parameter for name, parameter in self.named_parameters() if _filter(name)]
        decay_include = [parameter for name, parameter in self.named_parameters() if not _filter(name)]

        parameters = {'params': decay_exclude, 'weight_decay': 0.0},\
                     {'params': decay_include, 'weight_decay': self.weight_decay}
        optimizer = torch.optim.Adam(parameters,
                                     lr=self.learning_rate,
                                     betas=(self.beta_1, self.beta_2))

        if self.num_warmup_steps > 0:
            num_warmup_steps = self.num_warmup_steps

            def _learning_rate_scheduler(step: int) -> float:
                warmup_percent_done = step / num_warmup_steps
                # the learning rate should be reduce by step/warmup-step if in warmup-steps,
                # else the learning rate is fixed
                return min(1.0, warmup_percent_done)

            scheduler = LambdaLR(optimizer, _learning_rate_scheduler)

            schedulers = [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'strict': True,
                }
            ]
            return [optimizer], schedulers
        return optimizer


def _expand_sequence(inputs: torch.Tensor,
                     tokenizer: Tokenizer,
                     batch_first: bool = True
                     ) -> torch.Tensor:
    if tokenizer.pad_token is None:
        raise ValueError("This tokenizer does not have a padding token which is required for the BERT4Rec model.")
    input_shape = inputs.shape
    shape_addition_padding = (input_shape[0], 1)
    if not batch_first:
        shape_addition_padding = (1, input_shape[1])

    # generate a tensor with the addition seq step (filled with padding tokens)
    additional_padding = torch.full(shape_addition_padding, tokenizer.pad_token_id,
                                    dtype=inputs.dtype,
                                    device=inputs.device)
    if len(input_shape) > 2:
        additional_padding = additional_padding.repeat(1, input_shape[2]).unsqueeze(1)

    return torch.cat([inputs, additional_padding], dim=1 if batch_first else 0)


def _add_mask_token_at_ending(input_seq: torch.Tensor,
                              tokenizer: Tokenizer,
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ This methods adds the masking token at the position of the first padding token
        :return: the input_seq with the masking token,
    """
    input_seq = input_seq.clone()
    input_shape = input_seq.size()
    padding_input_to_use = input_seq
    if len(input_shape) > 2:
        padding_input_to_use = input_seq.max(dim=2).values
    padding_mask = get_padding_mask(padding_input_to_use, tokenizer, transposed=False)

    batch_size = input_shape[0]
    max_seq_length = input_shape[1]

    inverse_indices = torch.arange(start=max_seq_length,
                                   end=0,
                                   step=-1,
                                   device=input_seq.device).repeat([batch_size, 1])
    inverse_padding_positions = padding_mask * inverse_indices
    first_index = max_seq_length - inverse_padding_positions.max(dim=1).values
    target_mask = F.one_hot(first_index, max_seq_length).bool()
    input_seq[target_mask] = tokenizer.mask_token_id
    return input_seq, target_mask


# TODO: implement as collate function
def _mask_items(inputs: torch.Tensor,
                tokenizer: Tokenizer,
                mask_probability: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked items inputs/target for masked modeling. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask item which is necessary for masked modeling."
        )

    target = inputs.clone()
    # we sample a few items in all sequences for the mask training
    device_to_use = inputs.device
    target_shape_to_use = target.shape[:2]
    probability_matrix = torch.full(size=target_shape_to_use,
                                    fill_value=mask_probability,
                                    device=device_to_use)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,
                                                 dtype=torch.bool,
                                                 device=device_to_use),
                                    value=0.0)
    if tokenizer.pad_token is not None:
        padding_tensor_to_use = target
        if len(target.size()) > 2:
            # we can use the max function to check if each entry is the padding token
            padding_tensor_to_use = target.max(dim=2).values

        padding_mask = padding_tensor_to_use.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    # to compute loss on masked items, we set ignore index on the other positions
    target[~masked_indices] = CROSS_ENTROPY_IGNORE_INDEX

    # 80% of the time, we replace masked input items with mask item ([MASK])
    indices_replaced = torch.bernoulli(torch.full(target_shape_to_use, 0.8, device=device_to_use)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input items with random items
    indices_random = torch.bernoulli(torch.full(target_shape_to_use, 0.5,
                                                device=device_to_use)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), target.shape, dtype=torch.long, device=device_to_use)
    inputs[indices_random] = random_words[indices_random]

    # the rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, target


def get_padding_mask(tensor: torch.Tensor,
                     tokenizer: Tokenizer,
                     transposed: bool = True,
                     inverse: bool = False) -> torch.Tensor:
    """
    generates the padding mask based on the tokenizer (by default batch first)
    :param tensor:
    :param tokenizer:
    :param transposed:
    :param inverse

    :return:
    """
    # the masking should be true where the padding token is set

    if len(tensor.size()) > 2:
        tensor = tensor.max(dim=2).values

    if inverse:
        padding_mask = tensor.ne(tokenizer.pad_token_id)
    else:
        padding_mask = tensor.eq(tokenizer.pad_token_id)

    if transposed:
        return padding_mask.transpose(0, 1)

    return padding_mask
