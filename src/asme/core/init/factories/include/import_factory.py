import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import List, Union, Any, Dict

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories.util import check_config_keys_exist
from asme.core.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.core.utils.logging import get_logger
import importlib

logger = get_logger(__name__)


@contextmanager
def push_python_path(path: Union[str,os.PathLike]):
    """
        Prepends the given path to `sys.path`.
        This method is intended to use with `with`, so after its usage, its value willbe removed from
        `sys.path`.

        see https://github.com/allenai/allennlp/blob/a63e28c24d091fb371011f644fec3ebd29bfbb7e/allennlp/common/util.py#L316
        """
    # In some environments, such as TC, it fails when sys.path contains a relative path, such as ".".
    path = Path(path).resolve()
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        # Better to remove by value, in case `sys.path` was manipulated in between.
        sys.path.remove(path)


def _import_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


class ImportFactory(ObjectFactory):

    CONFIG_KEY = "imports"
    TOP_LEVEL_MODULE_NAME = "asme"
    DEFAULT_MODULES = [".data.datasets.config", ".core.modules.config"]

    def __init__(self):
        super().__init__()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        if check_config_keys_exist(config, [self.CONFIG_KEY]):
            for name, plugin_config in config.get(self.CONFIG_KEY).items():
                if not ("path" in plugin_config and "module" in plugin_config):
                    return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION,
                                          f"Plugin '{name}' requires both 'path' and 'module' parameters to be present.")

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Union[Any, Dict[str, Any], List[Any]]:
        # Load default modules
        for default_module in self.DEFAULT_MODULES:
            importlib.import_module(default_module, self.TOP_LEVEL_MODULE_NAME)

        # Load all extra modules specified in the config
        imports = config.get_or_default(self.CONFIG_KEY, {})
        if len(imports) > 0:
            logger.info(f"Loading {len(imports)} extra modules.")
            for name, plugin_config in imports.items():
                path = plugin_config["path"]
                module = plugin_config["module"]
                try:
                    with push_python_path(path):
                        module = _import_module(module)
                        if module is None:
                            logger.critical(f"Failed to load plugin '{name}' at '{path}'. Does the path exists?")
                            exit(1)
                        logger.info(f"Successfully loaded plugin '{name}' at '{path}'.")
                except Exception as e:
                    logger.critical(f"Failed to load plugin '{name}' at '{path}' due to the following exception:", exc_info=e)
                    exit(1)

        return None

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.CONFIG_KEY]

    def config_key(self) -> str:
        return self.CONFIG_KEY