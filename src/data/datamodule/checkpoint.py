import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from asme.init.context import Context


@dataclass
class PreprocessingCheckpoint:
    step: int
    context: Context

    def save(self, path: Union[str, Path]):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Union[str, Path]) -> 'PreprocessingCheckpoint':
        with open(path, "rb") as f:
            return pickle.load(f)