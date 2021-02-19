from typing import Dict, Any

from config.factories.config import Config
from config.factories.object_factory import ObjectFactory


class Container:
    def __init__(self, context: Dict[str, Any]):
        self.context = context


# idea: remove processed config options from config -> put result in context
class ContainerBuilder:
    def __init__(self):
        self.handlers = []
        self.was_invoked = []

    def register_handler(self, handler: ObjectFactory):
        self.handlers.append(handler)

    def build(self, config: Dict[str, Any]):
        context = {}
        configuration = Config(config)

        remaining_handlers = list(self.handlers)
        # compare configuration to the last one after every loop, if no changes exist and all remaining factories that have not been applied are optional, all is well, otherwise its fucked up.
        while len(remaining_handlers) > 0:
            next_handler = remaining_handlers.pop()

            if next_handler.can_build(configuration, context):
                next_handler.build(configuration, context)
            else:
                remaining_handlers.insert(0, next_handler)

        return Container(context)


