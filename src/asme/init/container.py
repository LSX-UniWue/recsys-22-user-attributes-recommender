from typing import Dict, Any

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from asme.init.trainer_builder import TrainerBuilder
from asme.tokenization.tokenizer import Tokenizer


class Container:

    """
    a container class holding necessary objects for training, testing, prediction
    """

    def __init__(self,
                 objects: Dict[str, Any]
                 ):
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

    def tokenizers(self) -> Dict[str, Tokenizer]:
        tokenizers = {}
        for key, value in self._objects.items():
            if key.startswith('tokenizers.'):
                tokenizers[key.replace('tokenizers.', '')] = value
        return tokenizers

    def tokenizer(self, tokenizer_id: str) -> Tokenizer:
        return self._objects["tokenizers." + tokenizer_id]
