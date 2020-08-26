import copy
import dataclasses
import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from configs.utils.args_utils import get_config_from_args


@dataclass
class ModelConfig(object):
    """
    base model config
    """

    DATA_CLASS_METADATA_KEY_HELP = 'help'
    DATA_CLASS_METADATA_KEY_DEFAULT_VALUE = 'default_value'

    MODEL_CONFIG_CONFIG_FILE = 'config_file'

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--{}'.format(cls.MODEL_CONFIG_CONFIG_FILE), type=str, default=None,
                            help='path to a config file')

        for field in dataclasses.fields(cls):
            field_name = field.name
            help_info = 'the parameter {} to set'.format(field_name)
            if cls.DATA_CLASS_METADATA_KEY_HELP in field.metadata:
                help_info = field.metadata[cls.DATA_CLASS_METADATA_KEY_HELP]
            parser.add_argument('--{}'.format(field_name), type=field.type, default=None,
                                help=help_info)
        return parser

    @classmethod
    def from_args(cls, **kwargs):
        config_file = kwargs.get(ModelConfig.MODEL_CONFIG_CONFIG_FILE, None)
        if config_file is not None:
            return cls.from_json_file(Path(config_file))

        init_dict = {}
        for field in dataclasses.fields(cls):
            var_name = field.name
            value = get_config_from_args(kwargs, var_name, field.metadata.get(cls.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE))
            init_dict[var_name] = value

        return cls(**init_dict)

    @classmethod
    def from_json_file(cls,
                       json_file: Path):
        """Constructs a `Config` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        dict_obj = json.loads(text)
        return cls(**dict_obj)

    # properties
    item_voc_size: int = dataclasses.field(metadata={
        DATA_CLASS_METADATA_KEY_HELP: 'the item vocab size',
        DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 32
    })
    max_seq_length: int = dataclasses.field(metadata={
        DATA_CLASS_METADATA_KEY_HELP: 'the max sequence length',
        DATA_CLASS_METADATA_KEY_DEFAULT_VALUE: 16
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
