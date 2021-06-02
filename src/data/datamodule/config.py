import copy
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from asme.init.config import Config
from asme.init.context import Context
from data.datamodule.preprocessing import PreprocessingAction
from data.datamodule.unpacker import Unpacker


@dataclass
class DatasetPreprocessingConfig:
    name: str
    url: Optional[str]
    location: Path
    unpacker: Optional[Unpacker] = None
    preprocessing_actions: List[PreprocessingAction] = field(default_factory=[])
    context: Context = Context()


@dataclass
class AsmeDataModuleConfig:
    dataset_preprocessing_config: DatasetPreprocessingConfig
    data_sources_config: Config


class PreprocessingConfigProvider:

    def __init__(self, factory: Callable[..., DatasetPreprocessingConfig], **default_values):
        self.factory = factory
        self.default_values = default_values

    def __call__(self, **kwargs) -> DatasetPreprocessingConfig:
        arguments = copy.deepcopy(self.default_values)
        arguments.update(kwargs)
        params = inspect.signature(self.factory).parameters
        actual_arguments = {}
        for name, param in params.items():
            if name not in arguments and param.default == inspect.Parameter.empty:
                raise KeyError(f"Parameter '{name}' is absent and no default value was provided. "
                               f"However, '{name}' is required by this PreprocessingConfigProvider.")

            if param.default == inspect.Parameter.empty:
                actual_arguments[name] = arguments[name]
        return self.factory(**arguments)


DATASET_CONFIG_PROVIDERS = {

}


def register_preprocessing_config_provider(name: str, config_provider: PreprocessingConfigProvider, overwrite: bool = False):
    if name in DATASET_CONFIG_PROVIDERS and not overwrite:
        raise KeyError(f"A dataset config for key '{name}' is already registered and overwrite was set to false.")

    DATASET_CONFIG_PROVIDERS[name] = config_provider


def get_preprocessing_config_provider(name: str) -> Optional[PreprocessingConfigProvider]:
    return DATASET_CONFIG_PROVIDERS.get(name, None)
