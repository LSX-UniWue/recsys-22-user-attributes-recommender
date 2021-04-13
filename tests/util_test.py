from typing import List, Any


def assert_list_equal(list1: List[Any],
                      list2: List[Any]) -> None:
    """ compares two lists with each other and checks that each element in the first list is equal to
     the element in the second last at the same index """
    assert len(list1) == len(list2)
    assert all([a == b for a, b in zip(list1, list2)])
