from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

from asme.init.context import Context
from data.datamodule.preprocessing import PreprocessingAction
from data.datamodule.unpacker import Unpacker


@dataclass
class DatasetConfig:
    name: str
    url: Optional[str]
    location: Path
    unpacker: Optional[Unpacker] = None
    preprocessing_actions: List[PreprocessingAction] = field(default_factory=[])
    context: Context = Context()


@dataclass
class AsmeDataModuleConfig:
    datasetConfig: DatasetConfig
