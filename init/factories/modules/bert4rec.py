from typing import List

from data.collate import PadDirection
from init.config import Config
from init.context import Context
from init.factories.tokenizer.tokenizer_factory import TokenizerFactory
from init.factories.util import check_config_keys_exist
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from models.bert4rec.bert4rec_model import BERT4RecModel
from modules import BERT4RecModule


class Bert4RecModuleFactory(ObjectFactory):

    def __init__(self):
        super().__init__()
        self.model_factory = BERT4RecModelFactory()
        self.metrics_container_factory = None # TODO

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.model_factory.can_build(config, context)

    def build(self, config: Config, context: Context) -> BERT4RecModule:
        learning_rate = config.get_or_default('learning_rate', 0.001)
        beta_1 = config.get_or_default('beta_1', 0.99)
        beta_2 = config.get_or_default('beta_2', 0.998)
        weight_decay = config.get_or_default('weight_decay', 0.001)
        num_warmup_steps = config.get_or_default('num_warmup_steps', 10000)

        tokenizer = context.get(TokenizerFactory.KEY + '.item')
        padding_direction_str = config.get_or_default('padding_direction', PadDirection.RIGHT.value)
        padding_direction = PadDirection[padding_direction_str.upper()]

        metrics = self.metrics_container_factory.build(config, context)

        model = self.model_factory.build(config.get_config(self.model_factory.config_path()), context)
        return BERT4RecModule(model, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                              weight_decay=weight_decay, num_warmup_steps=num_warmup_steps, tokenizer=tokenizer,
                              pad_direction=padding_direction, metrics=metrics)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'bert4rec_module'


class BERT4RecModelFactory(ObjectFactory):

    CONFIG_KEY_REQUIRED = ['item_vocab_size', 'max_seq_length', 'num_transformer_heads',
                           'num_transformer_layers', 'transformer_hidden_size', 'transformer_dropout']

    def can_build(self, config: Config, context: Context) -> CanBuildResult:

        config_keys_exist = check_config_keys_exist(config, self.CONFIG_KEY_REQUIRED)
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> BERT4RecModel:
        item_vocab_size = config.get('item_vocab_size')
        max_seq_length = config.get('max_seq_length')
        num_transformer_heads = config.get('num_transformer_heads')
        num_transformer_layers = config.get('num_transformer_layers')
        transformer_hidden_size = config.get('transformer_hidden_size')
        transformer_dropout = config.get('transformer_dropout')

        projection_layer_type = config.get_or_default('project_layer_type', 'transpose_embedding')
        embedding_pooling_type = config.get('embedding_pooling_type')
        initializer_range = config.get_or_default('initializer_range', 0.02)

        return BERT4RecModel(item_vocab_size=item_vocab_size, max_seq_length=max_seq_length,
                             num_transformer_heads=num_transformer_heads, num_transformer_layers=num_transformer_layers,
                             transformer_hidden_size=transformer_hidden_size, transformer_dropout=transformer_dropout,
                             project_layer_type=projection_layer_type, embedding_pooling_type=embedding_pooling_type,
                             initializer_range=initializer_range)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['model']

    def config_key(self) -> str:
        return 'bert4rec_model'
