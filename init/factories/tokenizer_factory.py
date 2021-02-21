from typing import List, Any

from init.config import Config
from init.context import Context
from init.factories.dependencies import DependenciesTrait
from init.factories.multiple_elements_factory import MultipleElementsFactoryTrait
from init.factories.vocabulary_factory import VocabularyFactory
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from tokenization.tokenizer import Tokenizer


class TokenizersFactory(ObjectFactory, MultipleElementsFactoryTrait):
    """
    Builds all tokenizers within the `tokenizers` section.
    """

    KEY = "tokenizers"

    def __init__(self):
        super(TokenizersFactory, self).__init__()
        self.tokenizer_factory = TokenizerFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.can_build_elements(config, context, self.tokenizer_factory)

    def build(self, config: Config, context: Context) -> Any:
        return self.build_elements(config, context, self.tokenizer_factory)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY


class TokenizerFactory(ObjectFactory, DependenciesTrait):
    """
    Builds a single tokenizer entry inside the tokenizers section.
    """
    # (AD) this is special since we can have multiple tokenizers with individual keys/names.
    KEY = "tokenizer"
    SPECIAL_TOKENS_KEY = "special_tokens"

    def __init__(self):
        super(TokenizerFactory, self).__init__()
        self.add_dependency(VocabularyFactory())

    def can_build(self, config: Config, context: Context) -> CanBuildResult:

        dependencies_result = self.can_build_dependencies(config, context)
        if dependencies_result.type != CanBuildResultType.CAN_BUILD:
            return dependencies_result

        if not config.has_path([self.SPECIAL_TOKENS_KEY]):
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION, f"missing key <{self.SPECIAL_TOKENS_KEY}>")

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context):
        vocabulary = self.get_dependency(VocabularyFactory.KEY).build(config.get_config([VocabularyFactory.KEY]), context)
        special_tokens = self._get_special_tokens(config)

        return Tokenizer(vocabulary, **special_tokens)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY

    def _get_special_tokens(self, config: Config):
        special_tokens_config = config.get_or_default([self.SPECIAL_TOKENS_KEY], {})
        return special_tokens_config
