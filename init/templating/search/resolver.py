from typing import Dict, Any, List

from optuna import Trial


def key_path_to_str(key_path: List[str]) -> str:
    return '.'.join(key_path)


class ParameterInfo:

    def __init__(self,
                 key_path: List[str],
                 suggest: str,
                 suggest_parameters: Dict[str, Any],
                 depends_on: str,
                 dependency: str):
        super().__init__()
        self.key_path = key_path
        self.suggest = suggest
        self.suggest_parameters = suggest_parameters
        self.depends_on = depends_on
        self.dependency = dependency

    @property
    def parameter_key(self) -> str:
        return key_path_to_str(self.key_path)


class ParameterResolver:
    def resolve(self, parameter_config: ParameterInfo, resolved_values: Dict[str, Any]) -> Any:
        raise NotImplementedError()

    def can_resolve(self, key: str) -> bool:
        raise NotImplementedError


class OptunaParameterResolver(ParameterResolver):

    OPTUNA_KEY: str = "hyper_opt"

    def __init__(self, trial: Trial):
        self.trial = trial

    def resolve(self, parameter_config: ParameterInfo, resolved_values: Dict[str, Any]) -> Any:
        function_name = 'suggest_' + parameter_config.suggest

        if not hasattr(self.trial, function_name):
            raise Exception(f"{function_name} is not a valid function name. Please refer to the documentation of the"
                            f" `Trial` object for valid names and arguments.")

        function_obj = getattr(self.trial, function_name)
        function_args = parameter_config.suggest_parameters.copy()
        function_args['name'] = parameter_config.parameter_key

        # TODO (AD) we could also validate that the provided arguments are sufficient
        # for calling the function and generate a meaningful error message
        value = function_obj(**function_args)

        depends_on = parameter_config.depends_on
        if not depends_on:
            return value

        dependency = parameter_config.dependency
        dependency_func = {
            'multiply': lambda x, y: x * y
        }[dependency]
        dependent_value = resolved_values.get(depends_on)

        return dependency_func(value, dependent_value)

    def can_resolve(self, key: str) -> bool:
        return key == self.OPTUNA_KEY


def parse_parameter_dependency_info(current_key: List[str],
                                    value: Dict[str, Any]
                                    ) -> ParameterInfo:
    """
    parses parameter infos from the dict
    :param current_key:
    :param value:
    :return: the parameter info object
    """
    key_path = current_key[:-1]  # here we remove model hyper_opt at the end
    suggest_func = value['suggest']
    suggest_params = value['params']
    depends_on = value.get('depends_on', None)
    dependency = value.get('dependency')

    if dependency is None and depends_on:
        raise ValueError(f'no dependency defined for {key_path_to_str(key_path)}')

    return ParameterInfo(key_path, suggest_func, suggest_params, depends_on, dependency)
