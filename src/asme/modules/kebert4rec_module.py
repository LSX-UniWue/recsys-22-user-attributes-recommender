import torch

from typing import Dict, List

from data.datasets import ITEM_SEQ_ENTRY_NAME
from asme.metrics.container.metrics_container import MetricsContainer
from asme.modules.util.module_util import get_padding_mask
from asme.tokenization.tokenizer import Tokenizer
from asme.models.kebert4rec.kebert4rec_model import KeBERT4RecModel
from asme.modules.bert4rec_module import BERT4RecBaseModule
from asme.utils.hyperparameter_utils import save_hyperparameters


class KeBERT4RecModule(BERT4RecBaseModule):

    """
    the module to train a keBERT4RecModel
    """

    @save_hyperparameters
    def __init__(self,
                 model: KeBERT4RecModel,
                 item_tokenizer: Tokenizer,
                 metrics: MetricsContainer,
                 additional_attributes: List[str],
                 learning_rate: float = 0.001,
                 beta_1: float = 0.99,
                 beta_2: float = 0.998,
                 weight_decay: float = 0.001,
                 num_warmup_steps: int = 10000
                 ):
        super().__init__(model=model,
                         item_tokenizer=item_tokenizer,
                         metrics=metrics,
                         learning_rate=learning_rate,
                         beta_1=beta_1,
                         beta_2=beta_2,
                         weight_decay=weight_decay,
                         num_warmup_steps=num_warmup_steps)
        self.attributes = additional_attributes

    def _forward_internal(self,
                          batch: Dict[str, torch.Tensor],
                          batch_idx: int
                          ) -> torch.Tensor:
        input_seq = batch[ITEM_SEQ_ENTRY_NAME]

        position_ids = KeBERT4RecModule.get_position_ids(batch)

        attribute_sequences = self._get_attribute_sequences(batch)

        # calc the padding mask
        padding_mask = get_padding_mask(input_seq, tokenizer=self.item_tokenizer)
        return self.model(input_seq, padding_mask=padding_mask, position_ids=position_ids, **attribute_sequences)

    def _get_attribute_sequences(self,
                                batch: Dict[str, torch.Tensor]
                                ) -> Dict[str, torch.Tensor]:
        return {
            attribute: batch[attribute] for attribute in self.attributes
        }
