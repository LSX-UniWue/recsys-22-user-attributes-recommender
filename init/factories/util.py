from typing import List

from init.config import Config
from init.context import Context


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
