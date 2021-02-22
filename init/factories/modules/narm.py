from typing import List

from init.config import Config
from init.context import Context
from init.factories.metrics.metrics_container import MetricsContainerFactory
from init.factories.tokenizer.tokenizer_factory import TokenizerFactory
from init.factories.util import check_config_keys_exist
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from models.narm.narm_model import NarmModel
from modules.narm_module import NarmModule


class NarmModuleFactory(ObjectFactory):

    def __init__(self):
        super().__init__()
        self.model_factory = NarmModelFactory()
        self.metrics_container_factory = MetricsContainerFactory()

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.model_factory.can_build(config.get_config(self.model_factory.config_path()), context)

    def build(self, config: Config, context: Context) -> NarmModule:
        learning_rate = config.get_or_default('learning_rate', 0.001)
        beta_1 = config.get_or_default('beta_1', 0.99)
        beta_2 = config.get_or_default('beta_2', 0.998)
        tokenizer = context.get(TokenizerFactory.KEY + '.item')

        metrics = self.metrics_container_factory.build(config.get_config(self.metrics_container_factory.config_path()),
                                                       context=context)
        model = self.model_factory.build(config.get_config(self.model_factory.config_path()), context)
        return NarmModule(model=model, tokenizer=tokenizer, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                          metrics=metrics)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'narm_module'


class NarmModelFactory(ObjectFactory):

    # FIXME: rename num_items => item_vocab_size
    CONFIG_KEY_REQUIRED = ['num_items', 'item_embedding_size', 'global_encoder_size',
                           'global_encoder_num_layers', 'embedding_dropout', 'context_dropout',
                           'embedding_pooling_type']

    def can_build(self, config: Config, context: Context) -> CanBuildResult:

        config_keys_exist = check_config_keys_exist(config, self.CONFIG_KEY_REQUIRED)
        if not config_keys_exist:
            return CanBuildResult(CanBuildResultType.MISSING_CONFIGURATION)

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> NarmModel:
        item_vocab_size = config.get('num_items')
        item_embedding_size = config.get('item_embedding_size')
        global_encoder_size = config.get('global_encoder_size')
        global_encoder_num_layers = config.get('global_encoder_num_layers')
        embedding_dropout = config.get('embedding_dropout')
        context_dropout = config.get('context_dropout')
        embedding_pooling_type = config.get('embedding_pooling_type')

        return NarmModel(num_items=item_vocab_size, item_embedding_size=item_embedding_size,
                         global_encoder_size=global_encoder_size, global_encoder_num_layers=global_encoder_num_layers,
                         embedding_dropout=embedding_dropout, context_dropout=context_dropout,
                         embedding_pooling_type=embedding_pooling_type)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return ['model']

    def config_key(self) -> str:
        return 'narm_model'
