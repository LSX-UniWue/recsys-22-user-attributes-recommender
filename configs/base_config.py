import dataclasses
import json
import logging
from abc import abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Any

from configs.utils.args_utils import get_config_from_args


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BaseConfig(object):
    """
    class that handles all
    TODO: rethink the inheritance
    """

    DATA_CLASS_METADATA_KEY_HELP = 'help'
    DATA_CLASS_METADATA_KEY_DEFAULT_VALUE = 'default_value'

    @classmethod
    @abstractmethod
    def get_config_file_key(cls) -> str:
        pass

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser) -> ArgumentParser:
        """
        adds model specific args
        :param parent_parser:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--{}'.format(cls.get_config_file_key()), type=str, default=None,
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
    def from_args(cls,
                  config_dict: Dict[str, Any],
                  **kwargs):
        """
        constructs a config from the line argument parser, and overrides it with the provided properties in the kwargs
        :param config_dict: the dictionary of the argument parser
        :param kwargs: all properties that should be overriden
        :return:
        """
        config_file = config_dict.get(cls.MODEL_CONFIG_CONFIG_FILE, None)
        if config_file is not None:
            config = cls.from_json_file(Path(config_file))
            properties_to_override = cls._get_config_dict_from_arguments(config_dict, use_default_value=False)
            config = cls._override_from_dict(config, properties_to_override)
        else:
            init_dict = cls._get_config_dict_from_arguments(config_dict, use_default_value=True)
            config = cls(**init_dict)

        return cls._override_from_dict(config, kwargs)

    @classmethod
    def _get_config_dict_from_arguments(cls,
                                        config_dict: Dict[str, Any],
                                        use_default_value: bool = False
                                        ) -> Dict[str, Any]:
        """
        builds a config dict from the other config dict based on the properties of the config class.
        (Note: It ignores all other entries of the config_dict that are not properties of the config class)
        :param config_dict: the config dict
        :param use_default_value: if True the default value of the property is used
        :return: a dict containing all relevant properties from the config dict for the config class
        """
        init_dict = {}
        for field in dataclasses.fields(cls):
            var_name = field.name
            default_value = None
            if use_default_value:
                default_value = field.metadata.get(BaseConfig.DATA_CLASS_METADATA_KEY_DEFAULT_VALUE)
            value = get_config_from_args(config_dict, var_name,
                                         default_value=default_value)
            if value is not None:
                init_dict[var_name] = value
        return init_dict

    @staticmethod
    def _override_from_dict(config,
                            kwargs_dict: Dict[str, Any]):
        """
        overrides the config properties with the specified values in the dict
        :param config: the config to override
        :param kwargs_dict: the dict with all values to override
        :return:
        """
        to_remove = []
        for key, value in kwargs_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs_dict.pop(key, None)

        if len(kwargs_dict) > 0:
            logger.warning('some arguments could not be set: {}'.format(','.join(kwargs_dict.keys())))

        return config

    @classmethod
    def from_json_file(cls,
                       json_file: Path):
        """
        Constructs a config from a json file of parameters.
        :param json_file: the path to the json file
        :return: the parsed config
        """
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        dict_obj = json.loads(text)
        return cls(**dict_obj)
