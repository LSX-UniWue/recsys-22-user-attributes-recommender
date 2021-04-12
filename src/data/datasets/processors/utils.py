
import torch


def random_uniform(start: float = 0., end: float = 1.) -> float:
    """
    Draws a single random number uniformly from a continouus distribution (pytorch) in [start; end).

    :param start: lowest number
    :param end: highest number

    :return: a single float from [start; end).
    """
    return torch.empty((), dtype=torch.float, device="cpu").uniform_().item()


def random_(start: int, end: int) -> int:
    """
    Draws uniformly from a discrete distribution in [start; end]

    :param start: lowest number.
    :param end: highest number.

    :return: a single number.
    """
    return torch.empty((), dtype=torch.int, device="cpu").random_(start, end).item()
