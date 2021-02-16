import abc
from typing import Dict, Any


class ObjectFactory:
    """
    Interface for factories that can participate in building objects from the configuration file.
    """
    @abc.abstractmethod
    def can_build(self, config: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Checks if the factory can build its object, based on the configuration and current context.
        This method is expected to raise an Exception if the constraints can never be satisfied, e.g. some configuration
        is missing.

        :param config: the configuration.
        :param context: the current context.
        :return: :code true if the object can be built, :code false otherwise.
        """
        pass

    @abc.abstractmethod
    def build(self, config: Dict[str, Any], context: Dict[str, Any]):
        """
        Builds the object and adds it to the context.

        :param config: the configuration.
        :param context: the current context.

        """
        pass