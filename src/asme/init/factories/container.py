from typing import List

from asme.init.config import Config
from asme.init.container import Container
from asme.init.context import Context
from asme.init.factories.common.conditional_based_factory import ConditionalFactory
from asme.init.factories.common.dependencies_factory import DependenciesFactory
from asme.init.factories.data_sources.data_sources import DataSourcesFactory
from asme.init.factories.modules.modules import GenericModuleFactory
from asme.init.factories.tokenizer.tokenizer_factory import TokenizersFactory
from asme.init.factories.trainer import TrainerBuilderFactory
from asme.init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType
from asme.models.basket.nnrec.nnrec_model import NNRecModel
from asme.models.bert4rec.bert4rec_model import BERT4RecModel
from asme.models.caser.caser_model import CaserModel
from asme.models.cosrec.cosrec_model import CosRecModel
from asme.models.narm.narm_model import NarmModel
from asme.models.rnn.rnn_model import RNNModel
from asme.models.sasrec.sas_rec_model import SASRecModel
from asme.models.kebert4rec.kebert4rec_model import KeBERT4RecModel
from asme.models.hgn.hgn_model import HGNModel
from asme.modules.baselines.bpr_module import BprModule
from asme.modules.baselines.markov_module import MarkovModule
from asme.modules.baselines.pop_module import PopModule
from asme.modules.baselines.session_pop_module import SessionPopModule
from asme.modules.basket.dream_module import DreamModule
from asme.modules.basket.nnrec_module import NNRecModule
from asme.modules.cosrec_module import CosRecModule
from asme.modules.masked_training_module import MaskedTrainingModule
from asme.modules.hgn_module import HGNModule
from asme.modules.next_item_prediction_training_module import NextItemPredictionTrainingModule
from asme.modules.sequence_next_item_prediction_training_module import SequenceNextItemPredictionTrainingModule


class ContainerFactory(ObjectFactory):
    def __init__(self):
        super().__init__()
        self.tokenizers_factory = TokenizersFactory()
        self.dependencies = DependenciesFactory(
            [
                ConditionalFactory('type', {'kebert4rec': GenericModuleFactory(MaskedTrainingModule, KeBERT4RecModel),
                                            'bert4rec': GenericModuleFactory(MaskedTrainingModule, BERT4RecModel),
                                            'caser': GenericModuleFactory(SequenceNextItemPredictionTrainingModule, CaserModel),
                                            'narm': GenericModuleFactory(NextItemPredictionTrainingModule, NarmModel),
                                            'sasrec': GenericModuleFactory(SequenceNextItemPredictionTrainingModule, SASRecModel),
                                            'rnn': GenericModuleFactory(NextItemPredictionTrainingModule, RNNModel),
                                            'cosrec': GenericModuleFactory(CosRecModule, CosRecModel),
                                            'hgn': GenericModuleFactory(HGNModule, HGNModel),
                                            'dream': GenericModuleFactory(DreamModule, RNNModel),
                                            'nnrec': GenericModuleFactory(NNRecModule, NNRecModel),
                                            'pop': GenericModuleFactory(PopModule, None),
                                            'session_pop': GenericModuleFactory(SessionPopModule, None),
                                            'markov': GenericModuleFactory(MarkovModule, None),
                                            'bpr': GenericModuleFactory(BprModule, None),},
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
