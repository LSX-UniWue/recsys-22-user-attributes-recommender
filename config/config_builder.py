from typing import Dict, Any

from config.factories.configuration import Configuration
from config.factories.object_factory import ObjectFactory


class Container:
    def __init__(self, context: Dict[str, Any]):
        self.context = context


class ContainerBuilder:
    def __init__(self):
        self.handlers = []

    def register_handler(self, handler: ObjectFactory):
        self.handlers.append(handler)

    def build(self, config: Dict[str, Any]):
        context = {}
        configuration = Configuration(config)

        remaining_handlers = list(self.handlers)

        while len(remaining_handlers) > 0:
            next_handler = remaining_handlers.pop()

            if next_handler.can_build(configuration, context):
                next_handler.build(configuration, context)
            else:
                remaining_handlers.insert(0, next_handler)

        return Container(context)


