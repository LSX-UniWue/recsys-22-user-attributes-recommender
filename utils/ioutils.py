from pathlib import Path
from typing import List, Callable, Any


def load_file_with_item_ids(path: Path) -> List[int]:
    """
    loads a file containing item ids into a list
    :param path: the path of the file
    :return:
    """
    items = _load_file_line_my_line(path, int)
    sorted(items)
    return items


def _load_file_line_my_line(path: Path, line_converter: Callable[[str], Any]) -> List[Any]:
    with open(path) as item_file:
        return [line_converter(line) for line in item_file.readlines()]
