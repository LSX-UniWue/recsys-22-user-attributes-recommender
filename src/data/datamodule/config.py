from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from data.datamodule.preprocessing import PreprocessingAction
from data.datamodule.unpacker import Unpacker


@dataclass
class DatasetConfig:
    name: str
    url: str
    location: Path
    unpacker: Optional[Unpacker] = None
    preprocessing_actions: List[PreprocessingAction] = field(default_factory=[])

@dataclass
class AsmeDataModuleConfig:
    datasetConfig: DatasetConfig
