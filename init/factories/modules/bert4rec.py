from typing import List, Union, Any, Dict

from init.config import Config
from init.context import Context
from init.object_factory import ObjectFactory, CanBuildResult
from modules import BERT4RecModule


class Bert4RecModuleFactory(ObjectFactory):
    def can_build(self, config: Config, context: Context) -> CanBuildResult:
        pass

    def build(self, config: Config, context: Context) -> BERT4RecModule:

        return BERT4RecModule(model, )

    def is_required(self, context: Context) -> bool:
        return True

    def config_path(self) -> List[str]:
        return []

    def config_key(self) -> str:
        return 'bert4rec_module'
