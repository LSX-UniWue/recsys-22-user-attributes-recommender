from dataclasses import dataclass, field

from configs.base_config import BaseConfig
from configs.models.model_config import ModelConfig


@dataclass
class GRUConfig(ModelConfig):
    """Config for GRU model"""

    @classmethod
    def get_arg_group_name(cls) -> str:
        return "GRU Model"

    gru_hidden_size: int = field(
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'hidden size of the gru layer',
            BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 64
        }
    )

    gru_token_embedding_size: int = field(
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'token embedding size',
            BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 64
        }
    )

    gru_num_layers: int = field(
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'number of gru layers',
            BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 1
        }
    )
