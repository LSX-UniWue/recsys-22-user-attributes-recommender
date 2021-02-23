from typing import Dict, Any

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from runner.util.builder import TrainerBuilder
from tokenization.tokenizer import Tokenizer


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

    def trainer(self) -> TrainerBuilder:
        return self._objects["trainer"]

    def tokenizer(self, id: str) -> Tokenizer:
        return self._objects["tokenizers"][id]
