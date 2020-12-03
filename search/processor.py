import copy
from typing import Any, Dict, Union

from search.resolver import ParameterResolver


class ConfigTemplateProcessor:

    def __init__(self, resolver: ParameterResolver):
        self.resolver = resolver

    def process(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes arbitrary configuration templates and resolves the values for keys reported by the resolver.
        Other values are just copied.

        :param template: a model configuration.
        :return: a fully resolved configuration.
        """
        def _resolve_recursively(template: Dict[str, Any]) -> Union[Any, Dict[str, Any]]:
            """
            Recursively resolves a configuration template against the resolver. If a key named `optuna` is discovered,
            resolution of the value will be delegated to the resolver. Other keys are copied.

            :param template: a configuration template with `optuna` sections that need to be resolved.

            :return: a fully resolved configuration.
            """
            result = {}

            for key, value in template.items():
                if self.resolver.can_resolve(key):
                    result = self.resolver.resolve(value)
                elif not isinstance(value, dict):
                    result[key] = copy.deepcopy(value)
                else:
                    result[key] = _resolve_recursively(value)
            return result

        return _resolve_recursively(template)
