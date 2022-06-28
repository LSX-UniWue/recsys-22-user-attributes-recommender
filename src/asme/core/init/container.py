from typing import Dict, Any, List

from asme.core.evaluation.evaluation import BatchEvaluator
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from asme.core.init.trainer_builder import TrainerBuilder
from asme.core.tokenization.tokenizer import Tokenizer
from asme.core.writer.prediction.batch_prediction_writer import BatchEvaluationWriter
from asme.data.datamodule.datamodule import AsmeDataModule


class Container:

    """
    a container class holding necessary objects for training, testing, prediction
    """

    def __init__(self,
                 objects: Dict[str, Any]
                 ):
        self._objects = objects

    def datamodule(self) -> AsmeDataModule:
        return self._objects["datamodule"]

    def train_dataloader(self) -> DataLoader:
        return self.datamodule().train_dataloader()

    def validation_dataloader(self) -> DataLoader:
        return self.datamodule().val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.datamodule().test_dataloader()

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

    def evaluators(self) -> List[BatchEvaluator]:
        return self._objects["evaluation"]["evaluators"]

    def writer(self) -> BatchEvaluationWriter:
        return self._objects["evaluation"]["writer"]

