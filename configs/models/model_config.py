import dataclasses
from dataclasses import dataclass

from configs.base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """
    Base class of every model config
    every subclass must also be annotated with dataclass
    """

    MODEL_CONFIG_CONFIG_FILE = 'model_config_file'

    @classmethod
    def get_config_file_key(cls) -> str:
        return cls.MODEL_CONFIG_CONFIG_FILE

    # properties
    item_voc_size: int = dataclasses.field(metadata={
        BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the item vocab size',
        BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 32
    })
    max_seq_length: int = dataclasses.field(metadata={
        BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the max sequence length',
        BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 16
    })
