from typing import Dict, Any


def get_config_from_args(arg_dict: Dict,
                         key: str,
                         default_value: Any = None) -> Any:
    """
    returns the value in the dict, if the value is None in the dict and a default value is provided, it returns the
    default value
    :param arg_dict: the dict containing all args
    :param key: the key to retrieve
    :param default_value: the default value
    :return: the value in the the dict for the provided key, if none the default value if provided
    """
    value_in_dict = arg_dict.get(key, None)
    if default_value is not None and value_in_dict is None:
        return default_value
    return value_in_dict
