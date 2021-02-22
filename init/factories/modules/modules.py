from typing import Any, List

from init.config import Config
from init.context import Context
from init.factories.common.conditional_based_factory import ConditionalFactory
from init.factories.modules.bert4rec import BERT4RecModuleFactory
from init.object_factory import ObjectFactory, CanBuildResult


class ModuleFactory(ObjectFactory):

    KEY = "module"

    def __init__(self):
        super().__init__()

        self.module_factory = ConditionalFactory('type', {'bert4rec': BERT4RecModuleFactory()})

    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        return self.module_factory.can_build(config, context)

    def build(self, config: Config, context: Context) -> Any:
        return self.module_factory.build(config, context)

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return [self.KEY]

    def config_key(self) -> str:
        return self.KEY
