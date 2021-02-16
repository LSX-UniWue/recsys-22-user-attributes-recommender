from pathlib import Path
from typing import Dict, Any

from config.factories.configuration import Configuration
from config.factories.object_factory import ObjectFactory
from tokenization.tokenizer import Tokenizer
from tokenization.vocabulary import CSVVocabularyReaderWriter, Vocabulary


# TODO (AD) support multiple tokenizers
class TokenizerFactory(ObjectFactory):

    TOKENIZER_KEY = "tokenizer"
    VOCABULARY_KEY = "vocabulary"
    SPECIAL_TOKENS_KEY = "special_tokens"

    def can_build(self, config: Configuration, context: Dict[str, Any]) -> bool:
        return config.has_path([self.TOKENIZER_KEY])

    def build(self, config: Configuration, context: Dict[str, Any]):
        vocabulary = self._create_vocabulary(config)
        special_tokens = self._get_special_tokens(config)

        context["tokenizer"] = Tokenizer(vocabulary, **special_tokens)

    def _create_vocabulary(self, config: Configuration) -> Vocabulary:

        delimiter = config.get_or_default([self.TOKENIZER_KEY, self.VOCABULARY_KEY, "delimiter"], "\t")
        vocab_file = config.get_or_raise([self.TOKENIZER_KEY,
                                          self.VOCABULARY_KEY, "file"],
                                         f"<file> could not be found in vocabulary config section.")

        vocab_reader = CSVVocabularyReaderWriter(delimiter)

        with Path(vocab_file).open("r") as file:
            return vocab_reader.read(file)

    def _get_special_tokens(self, config: Configuration):
        special_tokens_config = config.get_or_default([self.TOKENIZER_KEY, self.SPECIAL_TOKENS_KEY], {})
        return special_tokens_config
