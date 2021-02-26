import copy
from typing import Any, Dict, Union, List

from init.templating.search.resolver import ParameterResolver, ParameterInfo, key_path_to_str
from init.templating.template_processor import TemplateProcessor


def _resolve_dependency(parameter_dependency_infos: List[ParameterInfo]) -> List[ParameterInfo]:
    """
    simple method to resolve parameter dependency
    FIXME: this method does not check for cicles, maybe also fails for complex dependencies

    the method reorders the parameter infos in such a way that all necessary dependencies of a parameter are before
    the parameter in the list

    :param parameter_dependency_infos:
    :return:
    """
    dependency_free_parameters = []
    dependency_lists = {}

    for parameter_dependency_info in parameter_dependency_infos:
        depends_on = parameter_dependency_info.depends_on
        if depends_on:
            dependency_list = dependency_lists.get(depends_on, [])
            dependency_list.append(parameter_dependency_info)
            dependency_lists[depends_on] = dependency_list
        else:
            dependency_free_parameters.append(parameter_dependency_info)

    resolved_dependencies = []

    for dependency_free_parameter in dependency_free_parameters:
        resolved_dependencies.append(dependency_free_parameter)

        def _add_dependencies_recursively(parameter_dependency_info: ParameterInfo) -> List[ParameterInfo]:
            result = []
            var_key = parameter_dependency_info.parameter_key
            if var_key in dependency_lists:
                dependencies = dependency_lists[var_key]
                for dependency in dependencies:
                    result.append(dependency)
                    result.extend(_add_dependencies_recursively(dependency))
            return result

        resolved_dependencies.extend(_add_dependencies_recursively(dependency_free_parameter))

    return resolved_dependencies


def _parse_parameter_dependency_info(current_key, value: Dict[str, Any]) -> ParameterInfo:
    key_path = current_key[:-1]  # here we remove model hyper_opt at the end
    suggest_func = value['suggest']
    suggest_params = value['params']
    depends_on = value.get('depends_on', None)
    dependency = value.get('dependency')

    if dependency is None and depends_on:
        raise ValueError(f'no dependency defined for {key_path_to_str(key_path)}')

    return ParameterInfo(key_path, suggest_func, suggest_params, depends_on, dependency)


class SearchTemplateProcessor(TemplateProcessor):

    def __init__(self, resolver: ParameterResolver):
        self.resolver = resolver

    def can_modify(self, config: Dict[str, Any]) -> bool:
        return True

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes arbitrary configuration templates and resolves the values for keys reported by the resolver.
        Other values are just copied.

        :param config: a model configuration.
        :return: a fully resolved configuration.
        """

        def _find_all_resolvable_parameters(template: Dict[str, Any], current_keys=None) -> List[ParameterInfo]:
            if current_keys is None:
                current_keys = []
            parameter_info = []
            for key, value in template.items():
                current_key = current_keys + [key]
                if self.resolver.can_resolve(key):
                    parameter_info.append(_parse_parameter_dependency_info(current_key, value))
                if isinstance(value, dict):
                    parameter_info.extend(_find_all_resolvable_parameters(value, current_key))
            return parameter_info

        all_resolveable_parameters = _find_all_resolvable_parameters(config)

        parameters_to_resolve = _resolve_dependency(all_resolveable_parameters)

        resolved_values = {}

        for parameter_to_resolve in parameters_to_resolve:
            value_key = parameter_to_resolve.parameter_key
            value = self.resolver.resolve(parameter_to_resolve, resolved_values)
            resolved_values[value_key] = value

        # now replace the resolver directives in the template
        def _replace_recursively(template: Dict[str, Any], key_path: List[str] = None) -> Union[Any, Dict[str, Any]]:
            """
            Recursively resolves a configuration template against the resolver. If a key named `optuna` is discovered,
            resolution of the value will be delegated to the resolver. Other keys are copied.

            :param template: a configuration template with `optuna` sections that need to be resolved.
            :param key_path: the current key path

            :return: a fully resolved configuration.
            """
            if key_path is None:
                key_path = []
            result = {}

            for key, value in template.items():
                current_key_path = key_path + [key]
                if self.resolver.can_resolve(key):
                    # we use only the keypath, we removed hyper_opt from the param info
                    value_to_resolve = key_path_to_str(key_path)
                    result = resolved_values[value_to_resolve]
                elif not isinstance(value, dict):
                    result[key] = copy.deepcopy(value)
                else:
                    result[key] = _replace_recursively(value, current_key_path)
            return result

        return _replace_recursively(config)
