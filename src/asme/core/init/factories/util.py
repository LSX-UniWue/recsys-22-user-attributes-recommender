import os
from asme.core.tokenization.tokenizer import Tokenizer
from typing import List, Union, Any, Callable

from asme.core.init.config import Config
from asme.core.init.context import Context
from asme.core.init.factories import BuildContext
from asme.core.init.object_factory import CanBuildResult, CanBuildResultType, ObjectFactory
from asme.core.utils.logging import get_root_logger


def build_with_subsection(factory: ObjectFactory,
                          build_context: BuildContext,
                          factory_func: Callable[[ObjectFactory, BuildContext], Any] =
                            lambda factory, build_context: factory.build(build_context)) -> Any:
    """
    Builds an object using a factory. First sets the build_context to the needes subsection, as determined by
    ObjectFactory.config_path(). Then the object is build and the build_context is reset afterwards.

    :param factory: a factory.
    :param build_context: the current build context.
    :param factory_func: a function `f(factory, build_context) -> Any` that builds the actual object.
    :return: the object produced by the factory.
    """
    sub_section = factory.config_path()
    build_context.enter_sections(sub_section)
    obj = factory_func(factory, build_context)
    build_context.leave_sections(sub_section)
    return obj

def can_build_with_subsection(factory: ObjectFactory,
                              build_context: BuildContext,
                              factory_func: Callable[[ObjectFactory, BuildContext], Any] =
                                lambda factory, build_context: factory.can_build(build_context)) -> CanBuildResult:
    """
    Builds an object using a factory. First sets the build_context to the needes subsection, as determined by
    ObjectFactory.config_path(). Then the object is build and the build_context is reset afterwards.

    :param factory: a factory.
    :param build_context: the current build context.
    :param factory_func: a function `f(factory, build_context) -> CanBuildResult` that evaluates whether the factory can build the object.
    :return: the object produced by the factory.
    """
    sub_section = factory.config_path()
    build_context.enter_sections(sub_section)
    obj = factory_func(factory, build_context)
    build_context.leave_sections(sub_section)
    return obj


def check_config_keys_exist(config: Config, keys: List[str]) -> bool:
    for key in keys:
        if not config.has_path([key]):
            return False

    return True


def check_context_entries_exists(context: Context, keys: List[str]) -> bool:
    for key in keys:
        if not context.get(key):
            return False

    return True


def require_config_keys(config: Config,
                        required_key: List[str]
                        ) -> CanBuildResult:
    config_key = set(config.get_keys())
    missing_keys = set(required_key) - config_key
    if len(missing_keys) > 0:
        return CanBuildResult(
            CanBuildResultType.MISSING_CONFIGURATION,
            f"Could not find all required keys (missing: {','.join(missing_keys)}) "
            f"in config (path: {'.'.join(config.base_path)})."
        )

    return CanBuildResult(CanBuildResultType.CAN_BUILD)


def require_config_field_equal(config: Config,
                               field_key: Union[str, List[str]],
                               required_field_value: Any
                               ) -> CanBuildResult:
    field_value = config.get(field_key)
    build_result_type = CanBuildResultType.CAN_BUILD if field_value == required_field_value else CanBuildResultType.MISSING_CONFIGURATION

    return CanBuildResult(build_result_type)


def infer_whole_path(config: Config, key: Union[str, List[str]], base_path: str, relative_path: str,
                     log_warning: bool = True) -> None:
    """
    Infers the path to be used for a specific config value based on the currently present value. If it does not exist, 
    it is set to point to the path `relative_path` relative to base_path`. If it exists and is not an absolute
    path, `relative_path` is appended. Otherwise the current value is left unchanged.
    
    :param config: The config to perform inference on.
    :param key: The key to the value that should be inferred.
    :param base_path: The base directory to use for inference if only a relative path was provided.
    :param relative_path: The relative_path to the append to the base_path.
    :param log_warning: If set to true, log a warning including the property that was changed as well as its old and
                        inferred value using the root logger.
    """
    if isinstance(key, str):
        key = [key]
    complete_key = ".".join(config.base_path + key)

    if config.has_path(key):
        # The key was found, so we only infer something if it is not an absolute path
        value = config.get(key)
        if not os.path.isabs(value):
            complete_path = os.path.join(base_path, value)
            if log_warning:
                msg = f"Property '{complete_key}' contains a relative path. Prepending the location of the dataset. " \
                      f"Current value: '{value}' -> new value: '{complete_path}'. "
                get_root_logger().warning(msg)
            config.set(key, complete_path)
    else:
        # The key is missing, so we completely infer it
        complete_path = os.path.join(base_path, relative_path)
        if log_warning:
            msg = f"Property '{complete_key}' was not found in the configuration. Inferring it to '{complete_path}'."
            get_root_logger().warning(msg)
        config.set(key, complete_path)


def infer_base_path(config: Config, key: Union[str, List[str]], base_path: str, log_warning: bool = True) -> None:
    """
        Infers the path to be used for a specific config value based on the currently present value. If it exists and is
        not an absolute path, `base_path` is prepended. Otherwise the current value is left unchanged.

        :param config: The config to perform inference on.
        :param key: The key to the value that should be inferred.
        :param base_path: The base directory to use for inference if only a relative path was provided.
        :param log_warning: If set to true, log a warning including the property that was changed as well as its old and
                            inferred value using the root logger.
        """
    if config.has_path(key):
        value = config.get(key)
        if not os.path.isabs(value):
            if isinstance(key, str):
                key = [key]
            complete_path = os.path.join(base_path, value)
            if log_warning:
                complete_key = ".".join(config.base_path + key)
                msg = f"Property '{complete_key}' contains a relative path. Prepending the location of the dataset. " \
                      f"Current value: '{value}' -> new value: '{complete_path}'. "
                get_root_logger().warning(msg)
            config.set(key, complete_path)


def get_all_tokenizers_from_context(context: Context) -> Dict[str, Tokenizer]:
    """
    returns a dict with all tokenizers loaded in the context
    :param context: the context to extract the tokenizers from
    :return: the dict containing only tokenizers in the context
    """
    return {
        key: value for key, value in context.as_dict().items() if isinstance(value, Tokenizer)
    }