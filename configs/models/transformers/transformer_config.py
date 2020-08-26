from dataclasses import dataclass, field

from configs.base_config import BaseConfig
from configs.models.model_config import ModelConfig


@dataclass
class TransformerConfig(ModelConfig):
    """
    base config for transformer based configuration
    """

    transformer_hidden_size: int = field(
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'hidden size of the transformer encoder',
            BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 64
        }
    )

    num_transformer_heads: int = field(
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the number of heads of the transformer encoder',
            BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 2
        }
    )

    num_transformer_layers: int = field(
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the number of transformer layer',
            BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 2
        }
    )

    transformer_dropout: float = field(
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the dropout of the transformer encoder',
            BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 0.5
        }
    )
