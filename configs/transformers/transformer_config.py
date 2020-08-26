from dataclasses import dataclass, field

from configs.config import ModelConfig


@dataclass
class TransformerConfig(ModelConfig):
    """
    base config for transformer based configuration
    """

    transformer_hidden_size: int = field(metadata={
        ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'hidden size of the transformer encoder',
        ModelConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 64
    })
    num_transformer_heads: int = field(metadata={
        ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'the number of heads of the transformer encoder',
        ModelConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 2
    })
    num_transformer_layers: int = field(metadata={
        ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'the number of transformer layer',
        ModelConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 2
    })
    transformer_dropout: float = field(metadata={
        ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'the dropout of the transformer encoder',
        ModelConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 0.5
    })
