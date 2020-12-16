from dependency_injector import containers
from pytorch_lightning import Trainer


class TrainerBuilder:

    def __init__(self, container: containers.Container):
        self.container = container

    def build(self) -> Trainer:
        pass
