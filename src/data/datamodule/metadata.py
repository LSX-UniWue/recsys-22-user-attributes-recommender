from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json

from asme.init.context import Context
from data import RATIO_SPLIT_PATH_CONTEXT_KEY, LOO_SPLIT_PATH_CONTEXT_KEY
from datasets.data_structures.dataset_metadata import DatasetMetadata


@dataclass_json
@dataclass
class DatasetMetadata:
    ratio_path: Optional[str] = None
    loo_path: Optional[str] = None

    @staticmethod
    def from_context(context: Context) -> 'DatasetMetadata':
        ratio_path = context.get(RATIO_SPLIT_PATH_CONTEXT_KEY)
        loo_path = context.get(LOO_SPLIT_PATH_CONTEXT_KEY)

        # We need to convert to strings so this class is JSON serializable
        if ratio_path is not None:
            ratio_path = str(ratio_path)
        if loo_path is not None:
            loo_path = str(loo_path)
        return DatasetMetadata(ratio_path, loo_path)
