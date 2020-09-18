from dataclasses import dataclass

from configs.models.transformers.transformer_config import TransformerConfig


@dataclass
class BERT4RecConfig(TransformerConfig):
    """Config for a BERT4Rec Model"""

    @classmethod
    def get_arg_group_name(cls) -> str:
        return "BERT4Rec Model"


