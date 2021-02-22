from typing import List

from init.config import Config
from init.container import Container
from init.context import Context
from init.factories.common.dependencies_factory import DependenciesFactory
from init.factories.common.union_factory import UnionFactory
from init.factories.data_sources.data_sources import DataSourcesFactory
from init.factories.modules.bert4rec import BERT4RecModuleFactory
from init.factories.tokenizer.tokenizer_factory import TokenizersFactory
from init.factories.trainer import TrainerFactory
from init.object_factory import ObjectFactory, CanBuildResult, CanBuildResultType


class ContainerFactory(ObjectFactory):
    def __init__(self):
        super(ContainerFactory, self).__init__()
        self.tokenizers_factory = TokenizersFactory()
        self.dependencies = DependenciesFactory(
            [
                UnionFactory([
                    BERT4RecModuleFactory()
                ], "module", ["module"]),
                DataSourcesFactory(),
                TrainerFactory()
            ]
        )

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        can_build_result = self.tokenizers_factory.can_build(config, context)

        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        can_build_result = self.dependencies.can_build(config, context)

        if can_build_result.type != CanBuildResultType.CAN_BUILD:
            return can_build_result

        return CanBuildResult(CanBuildResultType.CAN_BUILD)

    def build(self, config: Config, context: Context) -> Container:
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