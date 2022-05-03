from typing import List

from asme.core.init.config import Config
from asme.core.init.context import Context


class InjectionContext:

    def __init__(self, config: Config, context: Context):
        self.config = config
        self.context = context
        self.location = []

    def get_config(self) -> Config:
        return self.config

    def get_context(self) -> Context:
        return self.context

    def get_location(self) -> List[str]:
        return self.location

    def set_location(self, location: List[str]):
        self.location = location
