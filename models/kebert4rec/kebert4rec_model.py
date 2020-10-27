import torch

from models.bert4rec.bert4rec_model import BERT4RecBaseModel


class KeBERT4Rec(BERT4RecBaseModel):

    def __init__(self,
                 transformer_hidden_size: int,
                 num_transformer_heads: int,
                 num_transformer_layers: int,
                 item_vocab_size: int,
                 max_seq_length: int,
                 dropout: float):
        super().__init__(transformer_hidden_size=transformer_hidden_size,
                         num_transformer_heads=num_transformer_heads,
                         num_transformer_layers=num_transformer_layers,
                         dropout=dropout)

        self.item_vocab_size = item_vocab_size
        self.max_seq_length = max_seq_length + 1



    def _embed_input(self, input_sequence: torch.Tensor, position_ids: torch.Tensor, **kwargs):
        pass

    def _projection(self, dense: torch.Tensor):
        pass