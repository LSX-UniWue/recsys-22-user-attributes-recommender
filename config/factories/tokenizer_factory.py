from pathlib import Path
from typing import List, Any

from config.factories.config import Config
from config.factories.context import Context
from config.factories.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from tokenization.tokenizer import Tokenizer
from tokenization.vocabulary import CSVVocabularyReaderWriter


class TokenizersFactory(ObjectFactory):

    KEY = "tokenizers"

    def __init__(self):
        self.tokenizer_factory = TokenizerFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:

        tokenizers = config.get([])
        if len(tokenizers) == 0:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION, f"At least one tokenizer must be specified.")

        for name in tokenizers.keys():
            tokenizer_config = config.get_config([name])
            can_build_result = self.tokenizer_factory.can_build(tokenizer_config, context)

            if not can_build_result.type == CanBuildResultType.CAN_BUILD:
                return can_build_result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Any:
        tokenizers = config.get([])

        result = {}
        for name in tokenizers.keys():
            tokenizer_config = config.get_config([name])
            tokenizer = self.tokenizer_factory.build(tokenizer_config, context)

            result[name] = tokenizer

        return result

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]


# TODO (AD) support multiple tokenizers
class TokenizerFactory(ObjectFactory):

    # (AD) this is special since we can have multiple tokenizers with individual names. Keep in mind that the
    # TokenizersFactory will take care to place the tokenizers at the correct path in the context
    KEY = "tokenizer"
    SPECIAL_TOKENS_KEY = "special_tokens"

    def __init__(self):
        self.dependencies = {
            VocabularyFactory.KEY: VocabularyFactory()
        }

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        for path, factory in self.dependencies.items():
            if not config.has_path([path]) and factory.is_required(context):
                return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION, f"missing key <{path}>")

        if not config.has_path([self.SPECIAL_TOKENS_KEY]):
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION, f"missing key <{self.SPECIAL_TOKENS_KEY}>")

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context):
        vocabulary = self.dependencies[VocabularyFactory.KEY].build(config.get_config([VocabularyFactory.KEY]), context)
        special_tokens = self._get_special_tokens(config)

        return Tokenizer(vocabulary, **special_tokens)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def _get_special_tokens(self, config: Config):
        special_tokens_config = config.get_or_default([self.SPECIAL_TOKENS_KEY], {})
        return special_tokens_config


class VocabularyFactory(ObjectFactory):

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
