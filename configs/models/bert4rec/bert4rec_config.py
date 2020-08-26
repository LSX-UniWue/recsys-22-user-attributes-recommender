from dataclasses import dataclass

from configs.models.transformers import TransformerConfig


@dataclass
class BERT4RecConfig(TransformerConfig):
    """Config for a BERT4Rec Model"""
    pass
