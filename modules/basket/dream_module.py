from typing import Union, Dict, Optional

import torch.nn.functional as F

import pytorch_lightning as pl
import torch

from data.datasets import ITEM_SEQ_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, \
    NEGATIVE_SAMPLES_ENTRY_NAME, TARGET_ENTRY_NAME

from models.rnn.rnn_model import RNNSeqItemRecommenderModel
from modules.util.module_util import build_eval_step_return_dict, get_padding_mask
from tokenization.tokenizer import Tokenizer


# FIXME: maybe merge with RNNModule and make loss configurable
class DreamModule(pl.LightningModule):

    def __init__(self,
                 model: RNNSeqItemRecommenderModel,
                 tokenizer: Tokenizer,
                 learning_rate: float,
                 weight_decay: float
                 ):
        super().__init__()

        self.model = model

        self.tokenizer = tokenizer

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int
                      ) -> Optional[Union[torch.Tensor, Dict[str, Union[torch.Tensor, float]]]]:
        """
        Performs a training step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S, M),
            * `POSITIVE_SAMPLES_ENTRY_NAME`: a tensor of size (N, M) containing the next sequence items (pos examples)
            * `NEGATIVE_SAMPLES_ENTRY_NAME`: a tensor of size (N, M) containing a negative item (sampled)

        Where N is the batch size, S the max sequence length and M the max items per sequence step.

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: the total loss
        """
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        pos_items = batch[POSITIVE_SAMPLES_ENTRY_NAME]
        neg_items = batch[NEGATIVE_SAMPLES_ENTRY_NAME]

        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)

        logits = self.model(input_seq, padding_mask)

        loss = self._calc_loss(logits, pos_items, neg_items)
        return {
            'loss': loss
        }

    def _calc_loss(self,
                   logit: torch.Tensor,
                   pos_items: torch.Tensor,
                   neg_items: torch.Tensor
                   ) -> torch.Tensor:
        # bpr FIXME: check
        # we only use the last position as target, because the rnn only encodes the complete sequence
        padding_mask = (~ pos_items.eq(self.tokenizer.pad_token_id)).sum(-1).sum(-1) - 1
        target_mask = F.one_hot(padding_mask, num_classes=pos_items.size()[1]).to(torch.bool)

        pos_items = pos_items[target_mask]
        neg_items = neg_items[target_mask]
        pos_logits = logit.gather(1, pos_items)
        neg_logits = logit.gather(1, neg_items)

        mask = ~ pos_items.eq(self.tokenizer.pad_token_id)
        num_items = mask.sum()

        score = F.logsigmoid(pos_logits - neg_logits)
        score = score * mask

        return - score.sum() / num_items

    def validation_step(self,
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int
                        ) -> Dict[str, torch.Tensor]:
        """
        Performs a validation step on a batch of sequences and returns the overall loss.

        `batch` must be a dictionary containing the following entries:
            * `ITEM_SEQ_ENTRY_NAME`: a tensor of size (N, S, M),
            * `TARGET_ENTRY_NAME`: a tensor of size (N, M) with the target items,

        A padding mask will be generated on the fly, and also the masking of items

        Where N is the batch size, S the max sequence length and M the max items per sequence step.

        :param batch: the batch
        :param batch_idx: the batch number.
        :return: A dictionary with entries according to `build_eval_step_return_dict`.
        """
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]
        targets = batch[TARGET_ENTRY_NAME]

        padding_mask = get_padding_mask(input_seq, self.tokenizer, transposed=False, inverse=True)
        prediction = self.model(input_seq, padding_mask)

        mask = ~ targets.eq(self.tokenizer.pad_token_id)
        return build_eval_step_return_dict(input_seq, prediction, targets, mask=mask)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
