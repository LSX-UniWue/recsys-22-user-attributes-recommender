from dataclasses import dataclass

from configs.models.transformers.transformer_config import TransformerConfig


@dataclass
class SASRecConfig(TransformerConfig):
    """
    configuration for a SASRecModel
    """

    @classmethod
    def get_arg_group_name(cls) -> str:
        return "SASRec Model"


