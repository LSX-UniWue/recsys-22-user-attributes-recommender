from dataclasses import dataclass, field

from configs.base_config import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):

    MODEL_CONFIG_CONFIG_FILE = 'training_config_file'

    @classmethod
    def get_config_file_key(cls) -> str:
        return cls.MODEL_CONFIG_CONFIG_FILE

    batch_size: int = field(
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the batch size to train the model',
            BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 128
        }
    )


@dataclass
class OptimizerConfig(object):

    learning_rate: float = field(
        default=1e-3,
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the learning rate of the optimizer'
        }
    )


@dataclass
class AdamOptimizerConfig(object):

    beta_1: float = field(
        default=0.9,
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the beta1 of the adam optimizer'
        }
    )

    beta_2: float = field(
        default=0.999,
        metadata={
            BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the beta2 of the adam optimizer'
        }
    )
