from typing import Dict, Any

from optuna import Trial


class ParameterResolver:
    def resolve(self, parameter_config: Dict[str, Any]) -> Any:
        raise NotImplementedError()

    def can_resolve(self, key: str) -> bool:
        raise NotImplementedError


class OptunaParameterResolver(ParameterResolver):

    OPTUNA_KEY: str = "optuna"

    def __init__(self, trial: Trial):
        self.trial = trial

    def resolve(self, parameter_config: Dict[str, Any]) -> Any:
        keys = parameter_config.keys()

        if len(keys) != 1:
            raise Exception(f"Invalid configuration {parameter_config}. There should only be a single root entry.")

        function_name = list(keys)[0]

        if not hasattr(self.trial, function_name):
            raise Exception(f"{function_name} is not a valid function name. Please refer to the documentation of the `Trial` object for valid names and arguments.")

        function_obj = getattr(self.trial, function_name)
        function_args = parameter_config[function_name]

        #TODO (AD) we could also validate that the provided arguments are sufficient
        # for calling the function and generate a meaningful error message
        return function_obj(**function_args)

    def can_resolve(self, key: str) -> bool:
        return key == self.OPTUNA_KEY
