from typing import Dict, Any, List, Optional


class Configuration:
    """
    An object that controls access to the configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the configuration object.
        :param config: a dictionary that contains the configuration values.
        """
        self.config = config

    def _resolve_value(self, path: List[str]) -> Any:
        current_section = self.config

        path_length = len(path)

        for i, key in enumerate(path):
            if not isinstance(current_section, Dict) or key not in current_section:
                return None
            if i < path_length - 1:
                current_section = current_section[key]
            else:
                return current_section[key]

    def get_or_raise(self, path: List[str], exception_msg: str = None) -> Any:
        """
        Gets the value associated with a key or raises an exception if the key can not be found.

        :param path: a path.
        :param exception_msg: a message for the exception.
        :return: a value.
        """
        value = self._resolve_value(path)

        if value is None:
            raise Exception(exception_msg)

        return value

    def get_or_default(self, path: List[str], default: Any) -> Any:
        """
        Gets the value associated with a path or the default value if the key can not be found.

        :param path: a path.
        :param default: a default value.
        :return: a value.
        """
        value = self._resolve_value(path)

        if value is None:
            return default

        return value

    def get(self, path: List[str]) -> Optional[Any]:
        """
        Gets the value associated with a path. If the path does not exist within the configuration, None is returned.
        :param path: a path.
        :return: a value, or None if the path does not exist.
        """
        return self._resolve_value(path)

    def has_path(self, path: List[str]) -> bool:
        """
        Checks whether a path exists in the configuration.
        :param path: a path.
        :return: :code true if the path exists, :code false otherwise.
        """
        return self.get(path) is not None
