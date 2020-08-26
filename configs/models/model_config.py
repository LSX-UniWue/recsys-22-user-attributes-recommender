import copy
import dataclasses
import json
from dataclasses import dataclass

from configs.base_config import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """
    Base class of every model config
    every subclass must also be annotated with dataclass
    """

    # properties
    item_voc_size: int = dataclasses.field(metadata={
        BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the item vocab size',
        BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 32
    })
    max_seq_length: int = dataclasses.field(metadata={
        BaseConfig.DATA_CLASS_METADATA_KEY_HELP: 'the max sequence length',
        BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 16
    })

    def to_dict(self):
        """ Serializes the config to a python dict. """
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """ Serializes the config to a JSON string. """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save the config to a json file. """
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())
