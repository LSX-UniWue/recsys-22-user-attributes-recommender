from typing import Any, List, Dict, Union

from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class MultipleElementsFactoryTrait:
    """
    A trait that adds common methods for working with multiple named elements. Assumes that all elements can be built
    by the same factory.
    """

    def __init__(self):
        super(MultipleElementsFactoryTrait, self).__init__()

    def can_build_elements(self, config: Config, context: Context, elements_factory: ObjectFactory) -> CanBuildResult:
        """
        Checks whether all subsequent elements in the config can be built using the given factory.

        :param config: a configuration section containing multiple named elements.
        :param context: the current context.
        :param elements_factory: a factory that can build these elements.
        :return: a result.
        """
        actual_config = config.get([])
        if len(actual_config) == 0:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION, f"At least one element must be specified.")

        for name in actual_config.keys():
            element_config = config.get_config([name])
            can_build_result = elements_factory.can_build(element_config, context)

            if not can_build_result.type == CanBuildResultType.CAN_BUILD:
                return can_build_result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build_elements(self, config: Config, context: Context, elements_factory: ObjectFactory) -> Union[Any, Dict[str, Any], List[Any]]:
        """
        Builds every element in the configuration using the given factory.

        :param config: a configuration section containing multiple named elements.
        :param context: the current context.
        :param elements_factory: a factory that can build every element.

        :return: a dictionary with the element names pointing to the built objects.
        """
        elements = config.get([])

        result = {}
        for name in elements.keys():
            element_config = config.get_config([name])
            tokenizer = elements_factory.build(element_config, context)

            result[name] = tokenizer

        return result
