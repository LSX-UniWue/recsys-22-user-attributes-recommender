import torch
import pytorch_lightning as pl
import torch.nn.functional as F

from typing import Tuple, List, Union, Dict, Optional

from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, POSITION_IDS
from models.bert4rec.bert4rec_model_2 import BERT4RecModel2
from modules import LOG_KEY_VALIDATION_LOSS, LOG_KEY_TEST_LOSS, LOG_KEY_TRAINING_LOSS
from modules.constants import RETURN_KEY_PREDICTIONS, RETURN_KEY_TARGETS, RETURN_KEY_MASK
from modules.util.module_util import get_padding_mask, convert_target_to_multi_hot, build_eval_step_return_dict
from tokenization.tokenizer import Tokenizer
from models.bert4rec.bert4rec_model import BERT4RecModel

CROSS_ENTROPY_IGNORE_INDEX = -100


class BERT4RecModule(pl.LightningModule):

    @staticmethod
    def get_position_ids(batch: Dict[str, torch.Tensor]
                         ) -> Optional[torch.Tensor]:
        return batch[POSITION_IDS] if POSITION_IDS in batch else None

    def __init__(self,
                 model: BERT4RecModel2,
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

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        position_ids = BERT4RecModule.get_position_ids(batch)
        input_seq = _expand_sequence(inputs=input_seq, tokenizer=self.tokenizer)

        if self.batch_first:
            input_seq = input_seq.transpose(0, 1)
            if position_ids is not None:
                position_ids = position_ids.transpose(0, 1)

        # calc the padding mask
        padding_mask = get_padding_mask(tensor=input_seq,
                                        tokenizer=self.tokenizer,
                                        transposed=True)

        # random mask some items
        # FIXME: paper quote: we also produce samples that only mask the last item
        # in the input sequences during training.
        # how? TODO: check code!
        input_seq, target = _mask_items(inputs=input_seq,
                                        tokenizer=self.tokenizer,
                                        mask_probability=self.mask_probability)

        # call the model
        prediction_logits = self.model(input_seq, padding_mask=padding_mask, position_ids=position_ids)

        masked_lm_loss = self._calc_loss(prediction_logits, target)
        self.log(LOG_KEY_TRAINING_LOSS, masked_lm_loss, prog_bar=False)
        return {
            'loss': masked_lm_loss,

        }

    def _calc_loss(self,
                   prediction_logits: torch.Tensor,
                   target: torch.Tensor,
                   is_eval: bool = False
                   ) -> torch.Tensor:
        target_size = len(target.size())
        vocab_size = prediction_logits.size()[-1]
        is_basket_recommendation = target_size > 1 if is_eval else target_size > 2
        if is_basket_recommendation:
            target = convert_target_to_multi_hot(target, vocab_size, self.tokenizer.pad_token_id)
            loss_fnc = nn.BCEWithLogitsLoss()
            return loss_fnc(prediction_logits, target)

        # handle single item per sequence step
        loss_func = nn.CrossEntropyLoss(ignore_index=CROSS_ENTROPY_IGNORE_INDEX)

        flatten_predictions = prediction_logits.view(-1, vocab_size)
        flatten_targets = target.view(-1)
        return loss_func(flatten_predictions, flatten_targets)

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        return self._eval_epoch_step(batch, batch_idx)

    def _eval_epoch_step(self,
                         batch: Dict[str, torch.Tensor],
                         batch_idx: int,
                         is_test: bool = False
                         ) -> Dict[str, torch.Tensor]:
        # shorter to allow the masking token
        input_seq = _expand_sequence(inputs=batch[ITEM_SEQ_ENTRY_NAME],
                                     tokenizer=self.tokenizer)
        targets = batch[TARGET_ENTRY_NAME]

        position_ids = BERT4RecModule.get_position_ids(batch)

        # set the last non padding token to the mask token
        input_seq, target_mask = _add_mask_token_at_ending(input_seq, self.tokenizer)

        if self.batch_first:
            input_seq = input_seq.transpose(1, 0)
            target_mask = target_mask.transpose(1, 0)
            if position_ids is not None:
                position_ids = position_ids.transpose(1, 0)

        # after adding the mask token we can calculate the padding mask
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=True)

        # get predictions for all seq steps
        prediction = self.model(input_seq, padding_mask=padding_mask, position_ids=position_ids)
        # extract the relevant seq steps, where the mask was set, here only one mask per sequence steps exists
        prediction = prediction[target_mask]

        loss = self._calc_loss(prediction, targets, is_eval=True)
        self.log(LOG_KEY_TEST_LOSS if is_test else LOG_KEY_VALIDATION_LOSS, loss, prog_bar=True)

        # when we have multiple target per sequence step, we have to provide a mask for the paddings applied to
        # the target tensor
        mask = None if len(targets.size()) == 1 else ~ targets.eq(self.tokenizer.pad_token_id)

        for name, metric in self.metrics.items():
            step_value = metric(prediction, targets, mask=mask)
            self.log(name, step_value, prog_bar=True)
        # FIXME: readd, just for testing
        # return build_eval_step_return_dict(prediction, targets, mask=mask)

    # FIXME: copy paste code from sas rec module
    def validation_epoch_end(self,
                             outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
                             ) -> None:
        for name, metric in self.metrics.items():
            self.log(name, metric.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self._eval_epoch_step(batch, batch_idx, is_test=True)

    def test_epoch_end(self,
                       outputs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]
                       ) -> None:
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        #def _filter(name: str) -> bool:
        #    return name.endswith("bias") or 'norm1' in name or 'norm2' in name or 'layer_norm' in name

        #decay_exclude = [parameter for name, parameter in self.named_parameters() if _filter(name)]
        #decay_include = [parameter for name, parameter in self.named_parameters() if not _filter(name)]

        #parameters = {'params': decay_exclude, 'weight_decay': 0.0},\
        #             {'params': decay_include, 'weight_decay': self.weight_decay}

        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay,
                                     betas=(self.beta_1, self.beta_2))

        # if self.num_warmup_steps > 0:
        #     num_warmup_steps = self.num_warmup_steps
        #
        #     def _learning_rate_scheduler(step: int) -> float:
        #         warmup_percent_done = step / num_warmup_steps
        #         # the learning rate should be reduce by step/warmup-step if in warmup-steps,
        #         # else the learning rate is fixed
        #         return min(1.0, warmup_percent_done)
        #
        #     scheduler = LambdaLR(optimizer, _learning_rate_scheduler)
        #
        #     schedulers = [
        #         {
        #             'scheduler': scheduler,
        #             'interval': 'step',
        #             'strict': True,
        #         }
        #     ]
        #     return [optimizer], schedulers
        return optimizer


def _expand_sequence(inputs: torch.Tensor,
                     tokenizer: Tokenizer,
                     batch_first: bool = True
                     ) -> torch.Tensor:
    if tokenizer.pad_token is None:
        raise ValueError("This tokenizer does not have a padding token which is required for the BERT4Rec model.")
    input_shape = inputs.shape
    shape_addition_padding = (1, input_shape[1])
    if batch_first:
        shape_addition_padding = (input_shape[0], 1)

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


# TODO: implement as collate function (as a processor)
def _mask_items(inputs: torch.Tensor,
                tokenizer: Tokenizer,
                mask_probability: float
                ) -> Tuple[torch.Tensor, torch.Tensor]:
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
