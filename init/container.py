from typing import Dict, Any

from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader


class Container:
    def __init__(self, objects: Dict[str, Any]):
        self._objects = objects

    def train_dataloader(self) -> DataLoader:
        return self._objects["data_sources.train"]

    def validation_dataloader(self) -> DataLoader:
        return self._objects["data_sources.validation"]

    def test_dataloader(self) -> DataLoader:
        return self._objects["data_sources.test"]

    def module(self) -> LightningModule:
        return self._objects["module"]

    def trainer(self) -> Trainer:
        return self._objects["trainer"]
