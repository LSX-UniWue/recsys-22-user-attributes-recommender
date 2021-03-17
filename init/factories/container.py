from typing import List

from init.config import Config
from init.container import Container
from init.context import Context
from init.factories.common.conditional_based_factory import ConditionalFactory
from init.factories.common.dependencies_factory import DependenciesFactory
from init.factories.data_sources.data_sources import DataSourcesFactory
from init.factories.modules.modules import GenericModuleFactory
from init.factories.tokenizer.tokenizer_factory import TokenizersFactory
from init.factories.trainer import TrainerBuilderFactory
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from models.basket.nnrec.nnrec_model import NNRecModel
from models.bert4rec.bert4rec_model import BERT4RecModel
from models.caser.caser_model import CaserModel
from models.narm.narm_model import NarmModel
from models.rnn.rnn_model import RNNModel
from models.sasrec.sas_rec_model import SASRecModel
from models.kebert4rec.kebert4rec_model import KeBERT4RecModel
from models.hgn.hgn_model import HGNModel
from modules import BERT4RecModule, CaserModule, SASRecModule, KeBERT4RecModule, HGNModule
from modules.baselines.bpr_module import BprModule
from modules.baselines.markov_module import MarkovModule
from modules.baselines.pop_module import PopModule
from modules.baselines.session_pop_module import SessionPopModule
from modules.basket.dream_module import DreamModule
from modules.basket.nnrec_module import NNRecModule
from modules.narm_module import NarmModule
from modules.rnn_module import RNNModule


class ContainerFactory(ObjectFactory):
    def __init__(self):
        super(ContainerFactory, self).__init__()
        self.tokenizers_factory = TokenizersFactory()
        self.dependencies = DependenciesFactory(
            [
                ConditionalFactory('type', {'kebert4rec': GenericModuleFactory(KeBERT4RecModule, KeBERT4RecModel),
                                            'bert4rec': GenericModuleFactory(BERT4RecModule, BERT4RecModel),
                                            'caser': GenericModuleFactory(CaserModule, CaserModel),
                                            'narm': GenericModuleFactory(NarmModule, NarmModel),
                                            'sasrec': GenericModuleFactory(SASRecModule, SASRecModel),
                                            'rnn': GenericModuleFactory(RNNModule, RNNModel),
                                            'hgn': GenericModuleFactory(HGNModule, HGNModel),
                                            'dream': GenericModuleFactory(DreamModule, RNNModel),
                                            'nnrec': GenericModuleFactory(NNRecModule, NNRecModel),
                                            'pop': GenericModuleFactory(PopModule, None),
                                            'session_pop': GenericModuleFactory(SessionPopModule, None),
                                            'markov': GenericModuleFactory(MarkovModule, None),
                                            'bpr' : GenericModuleFactory(BprModule, None),},
                                   config_key='module',
                                   config_path=['module']),
                DataSourcesFactory(),
                TrainerBuilderFactory()
            ]
        )

    def can_build(self,
                  config: Config,
                  context: Context
                  ) -> CanBuildResult:
        tokenizer_config = config.get_config(self.tokenizers_factory.config_path())
        can_build_result = self.tokenizers_factory.can_build(tokenizer_config, context)

        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        can_build_result = self.dependencies.can_build(config, context)

        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self,
              config: Config,
              context: Context
              ) -> Container:
        # we need the tokenizers in the context because many objects have dependencies
        tokenizers_config = config.get_config(self.tokenizers_factory.config_path())

        tokenizers = self.tokenizers_factory.build(tokenizers_config, context)

        for key, tokenizer in tokenizers.items():
            path = list(tokenizers_config.base_path)
            path.append(key)
            context.set(path, tokenizer)

        all_dependencies = self.dependencies.build(config, context)

        for key, object in all_dependencies.items():
            if isinstance(object, dict):
                for section, o in object.items():
                    context.set([key, section], o)
            else:
                context.set(key, object)

        return Container(context.as_dict())

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return ""
