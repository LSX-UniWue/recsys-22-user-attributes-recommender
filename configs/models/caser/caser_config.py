from dataclasses import dataclass, field

from configs.models.model_config import ModelConfig
from models.layers.util_layers import ACTIVATION_FUNCTION_KEY_RELU


@dataclass
class CaserConfig(ModelConfig):

    user_voc_size: int = field(metadata={
        ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'the number of different users',
        ModelConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 8
    })

    d: int = field(metadata={
        ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'the embedding size',
        ModelConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 32
    })

    dropout: float = field(metadata={
        ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'the dropout rate of the model',
        ModelConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 0.5
    })

    num_horizontal_filters: int = field(metadata={
        ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'the number of horizontal filters',
        ModelConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 32
    })

    num_vertical_filters: int = field(metadata={
        ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'the number of vertical filters',
        ModelConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 32
    })

    # TODO: discuss: this is a training config

    num_negative_samples: int = field(metadata={
        ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'number of negative samples',
        ModelConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 3
    })

    conv_activation_fn: str = field(
        default=ACTIVATION_FUNCTION_KEY_RELU,
        metadata={
            ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'the activation function to use for the conv layer',
        }
    )

    fc_activation_fn: str = field(
        default=ACTIVATION_FUNCTION_KEY_RELU,
        metadata={
            ModelConfig.DATA_CLASS_METADATA_KEY_HELP: 'the activation function to use for the fc layer',
        }
    )