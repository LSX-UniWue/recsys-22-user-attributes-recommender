import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Union, Dict, Optional

from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from data.collate import PadDirection
from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME, POSITION_IDS
from modules import LOG_KEY_VALIDATION_LOSS, LOG_KEY_TEST_LOSS, LOG_KEY_TRAINING_LOSS
from modules.util.module_util import get_padding_mask, convert_target_to_multi_hot, build_eval_step_return_dict
from tokenization.tokenizer import Tokenizer
from models.bert4rec.bert4rec_model import BERT4RecModel


class BERT4RecModule(pl.LightningModule):

    """
    BERT4Rec module for the BERT4Rec model
    """

    @staticmethod
    def get_position_ids(batch: Dict[str, torch.Tensor]
                         ) -> Optional[torch.Tensor]:
        return batch[POSITION_IDS] if POSITION_IDS in batch else None

    def __init__(self,
                 model: BERT4RecModel,
                 mask_probability: float,
                 learning_rate: float,
                 beta_1: float,
                 beta_2: float,
                 weight_decay: float,
                 num_warmup_steps: int,
                 tokenizer: Tokenizer,
                 pad_direction: PadDirection
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
        self.pad_direction = pad_direction

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        """
        Performs a training step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
        Optional entries are:
            * `POSITION_IDS` a tensor of size (N, S) containing the position ids for the provided sequence

        Where N is the batch size and S the max sequence length.

        A padding mask will be generated on the fly, and also the masking of items

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: the total loss
        """

        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        target = batch[TARGET_ENTRY_NAME]
        position_ids = BERT4RecModule.get_position_ids(batch)
        input_seq = _expand_sequence(inputs=input_seq, padding_to_use=self.tokenizer.pad_token_id,
                                     pad_direction=self.pad_direction)
        target = _expand_sequence(inputs=target, padding_to_use=self.tokenizer.pad_token_id,
                                  pad_direction=self.pad_direction)

        # calc the padding mask
        padding_mask = get_padding_mask(tensor=input_seq,
                                        tokenizer=self.tokenizer,
                                        transposed=False,
                                        inverse=True)

        # call the model
        prediction_logits = self.model(input_seq, padding_mask=padding_mask, position_ids=position_ids)

        masked_lm_loss = self._calc_loss(prediction_logits, target)
        self.log(LOG_KEY_TRAINING_LOSS, masked_lm_loss, prog_bar=False)
        return {
            'loss': masked_lm_loss
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
        loss_func = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        flatten_predictions = prediction_logits.view(-1, vocab_size)
        flatten_targets = target.view(-1)
        return loss_func(flatten_predictions, flatten_targets)

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        """
        Performs a validation step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S),
            * `TARGET_ENTRY_NAME`: a tensor of size (N) with the target items,
        Optional entries are:
            * `POSITION_IDS` a tensor of size (N, S) containing the position ids for the provided sequence

        A padding mask will be generated on the fly, and also the masking of items

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.
        """

        return self._eval_epoch_step(batch, batch_idx)

    def _eval_epoch_step(self,
                         batch: Dict[str, torch.Tensor],
                         batch_idx: int,
                         is_test: bool = False
                         ) -> Dict[str, torch.Tensor]:
        original_input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]

        position_ids = BERT4RecModule.get_position_ids(batch)

        # set the last non padding token to the mask token
        input_seq = _add_mask_token_at_ending(original_input_seq, self.tokenizer, pad_direction=self.pad_direction)
        target_mask = input_seq.eq(self.tokenizer.mask_token_id)

        # handle basket training and evaluation
        if len(target_mask.size()) > 2:
            target_mask = target_mask.max(dim=-1)[0]

        # after adding the mask token we can calculate the padding mask
        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        # get predictions for all seq steps
        prediction = self.model(input_seq, padding_mask=padding_mask, position_ids=position_ids)
        # extract the relevant seq steps, where the mask was set, here only one mask per sequence steps exists
        prediction = prediction[target_mask]

        loss = self._calc_loss(prediction, targets, is_eval=True)
        self.log(LOG_KEY_TEST_LOSS if is_test else LOG_KEY_VALIDATION_LOSS, loss, prog_bar=True)

        # when we have multiple target per sequence step, we have to provide a mask for the paddings applied to
        # the target tensor
        mask = None if len(targets.size()) == 1 else ~ targets.eq(self.tokenizer.pad_token_id)
        return build_eval_step_return_dict(original_input_seq, prediction, targets, mask=mask)

    def test_step(self, batch, batch_idx):
        return self._eval_epoch_step(batch, batch_idx, is_test=True)

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
        return [optimizer]


def _expand_sequence(inputs: torch.Tensor,
                     padding_to_use: int,
                     pad_direction: PadDirection = PadDirection.RIGHT
                     ) -> torch.Tensor:
    input_shape = inputs.shape
    shape_addition_padding = (input_shape[0], 1)

    # generate a tensor with the addition seq step (filled with padding tokens)
    additional_padding = torch.full(shape_addition_padding, padding_to_use,
                                    dtype=inputs.dtype,
                                    device=inputs.device)
    if len(input_shape) > 2:
        additional_padding = additional_padding.repeat(1, input_shape[2]).unsqueeze(1)

    tensors_to_cat = [inputs, additional_padding] if pad_direction == PadDirection.RIGHT else [additional_padding, inputs]
    return torch.cat(tensors_to_cat, dim=1)


def _add_mask_token_at_ending(input_seq: torch.Tensor,
                              tokenizer: Tokenizer,
                              pad_direction: PadDirection = PadDirection.RIGHT
                              ) -> torch.Tensor:
    """
    This methods adds the masking token at the position of the first padding token
    :param input_seq: a batch first tensor of sequences
    :param tokenizer: the tokenizer to use
    :param pad_direction: the pad direction of the sequence
    :return: the input_seq with the masking token
    """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask item which is necessary for masked modeling."
        )

    # if the pad direction in left, we only have to add the mask token at the end of each sequence / batch
    if pad_direction == PadDirection.LEFT:
        mask_tokens = torch.full([input_seq.size()[0], 1], tokenizer.mask_token_id,
                                 dtype=input_seq.dtype,
                                 device=input_seq.device)
        expanded_input_seq = torch.cat([input_seq, mask_tokens], dim=1)
        return expanded_input_seq

    # if the pad direction is right, we first expand the sequence with a padding token
    input_seq = _expand_sequence(inputs=input_seq, padding_to_use=tokenizer.pad_token_id, pad_direction=pad_direction)

    # then we have to find the first index of the padding token
    input_shape = input_seq.size()
    padding_input_to_use = input_seq
    if len(input_shape) > 2:
        padding_input_to_use = input_seq.max(dim=2).values
    # here we calculate the token mask (inverted padding mask)
    token_mask = get_padding_mask(padding_input_to_use, tokenizer, transposed=False, inverse=True)
    # we calculate the length of the sequence, which is also the index of the first padding token
    first_padded_indices = token_mask.sum(dim=1)
    # convert the first_padding_indices to a boolean one hot vector
    target_mask = F.one_hot(first_padded_indices, num_classes=input_shape[1]).to(dtype=torch.bool)
    # set the first padding token to the mask token
    input_seq[target_mask] = tokenizer.mask_token_id

    return input_seq
