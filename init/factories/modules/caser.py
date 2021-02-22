from typing import List

from init.config import Config
from init.context import Context
from init.factories.metrics.metrics_container import MetricsContainerFactory
from init.factories.tokenizer.tokenizer_factory import TokenizerFactory
from init.factories.util import check_config_keys_exist
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from models.caser.caser_model import CaserModel
from modules import CaserModule


class CaserModuleFactory(ObjectFactory):

    def __init__(self):
        super().__init__()
        self.model_factory = CaserModelFactory()
        self.metrics_container_factory = MetricsContainerFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.model_factory.can_build(config.get_config(self.model_factory.config_path()), context)

    def build(self, config: Config, context: Context) -> CaserModule:
        learning_rate = config.get_or_default('learning_rate', 0.001)
        weight_decay = config.get_or_default('weight_decay', 0.001)
        tokenizer = context.get(TokenizerFactory.KEY + '.item')

        metrics = self.metrics_container_factory.build(config.get_config(self.metrics_container_factory.config_path()),
                                                       context=context)

        model = self.model_factory.build(config.get_config(self.model_factory.config_path()), context)
        return CaserModule(model=model, tokenizer=tokenizer, learning_rate=learning_rate, weight_decay=weight_decay,
                           metrics=metrics)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'caser_module'


class CaserModelFactory(ObjectFactory):

    CONFIG_KEY_REQUIRED = ['item_vocab_size', 'max_seq_length', 'embedding_size',
                           'user_vocab_size', 'num_vertical_filters', 'num_horizontal_filters',
                           'conv_activation_fn', 'fc_activation_fn', 'dropout']

    def can_build(self, config: Config, context: Context) -> CanBuildResult:

        config_keys_exist = check_config_keys_exist(config, self.CONFIG_KEY_REQUIRED)
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> CaserModel:
        item_vocab_size = config.get('item_vocab_size')
        max_seq_length = config.get('max_seq_length')
        embedding_size = config.get('embedding_size')
        user_vocab_size = config.get('user_vocab_size')
        num_vertical_filters = config.get('num_vertical_filters')
        num_horizontal_filters = config.get('num_horizontal_filters')

        conv_activation_fn = config.get('conv_activation_fn')
        embedding_pooling_type = config.get('embedding_pooling_type')
        fc_activation_fn = config.get('fc_activation_fn')
        dropout = config.get('dropout')

        return CaserModel(embedding_size=embedding_size, item_vocab_size=item_vocab_size,
                          user_vocab_size=user_vocab_size, max_seq_length=max_seq_length,
                          num_vertical_filters=num_vertical_filters, num_horizontal_filters=num_horizontal_filters,
                          conv_activation_fn=conv_activation_fn, fc_activation_fn=fc_activation_fn, dropout=dropout,
                          embedding_pooling_type=embedding_pooling_type)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['model']

    def config_key(self) -> str:
        return 'caser_model'
