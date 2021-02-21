from pathlib import Path
from typing import List

from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from tokenization.vocabulary import CSVVocabularyReaderWriter


class VocabularyFactory(ObjectFactory):
    """
    Builds a vocabulary.
    """
    KEY = "vocabulary"
    REQUIRED_KEYS = ["file", "delimiter"]

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        for key in self.REQUIRED_KEYS:
            if not config.has_path([key]):
                return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION, f"missing key <{key}>")

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context):
        delimiter = config.get_or_default(["delimiter"], "\t")
        vocab_file = config.get_or_raise(["file"], f"<file> could not be found in vocabulary config section.")

        vocab_reader = CSVVocabularyReaderWriter(delimiter)

        with Path(vocab_file).open("r") as file:
            return vocab_reader.read(file)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY