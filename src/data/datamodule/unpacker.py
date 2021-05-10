import zipfile
from abc import abstractmethod
from pathlib import Path


class Unpacker:

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def apply(self, location: Path) -> Path:
        pass

    def __call__(self, location: Path) -> Path:
        return self.apply(location)


class Unzipper(Unpacker):

    def __init__(self, target_directory: Path):
        self.target_directory = target_directory

    def name(self) -> str:
        return "Dataset Unzipper"

    def apply(self, location: Path) -> Path:
        with zipfile.ZipFile(location) as zip_file:
            zip_file.extractall(self.target_directory)

        return self.target_directory
