from typing import List, Union, Any

from init.config import Config
from init.context import Context
from init.object_factory import CanBuildResult, CanBuildResultType


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

    if not check_config_keys_exist(config, required_key):
        return CanBuildResult(
            CanBuildResultType.MISSING_CONFIGURATION,
            f"Could not find all required keys ({required_key}) in config."
        )

    return CanBuildResult(CanBuildResultType.CAN_BUILD)


def require_config_field_equal(config: Config,
                               field_key: Union[str, List[str]],
                               required_field_value: Any
                               ) -> CanBuildResult:
    field_value = config.get(field_key)
    build_result_type = CanBuildResultType.CAN_BUILD if field_value == required_field_value else CanBuildResultType.MISSING_CONFIGURATION

    return CanBuildResult(build_result_type)