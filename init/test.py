from init.config import Config
from init.config_builder import ContainerBuilder
from init.context import Context
from init.factories.data_sources.data_sources import DataSourcesFactory
from init.factories.modules.modules import ModuleFactory
from init.factories.tokenizer.tokenizer_factory import TokenizersFactory
from init.object_factory import CanBuildResultType

if __name__ == "__main__":
    import json
    import _jsonnet

    from pathlib import Path

    config_file = Path("../projects/examples/bert4rec_config.jsonnet")
    config_json = _jsonnet.evaluate_file(str(config_file))

    config = json.loads(config_json)
    print(config)

    builder = ContainerBuilder()

    tokenizers_factory = TokenizersFactory()
    data_sources_factory = DataSourcesFactory()
    module_factory = ModuleFactory()

    config = Config(config)
    context = Context()

    if not config.has_path(tokenizers_factory.config_path()):
        print(f"Missing tokenizer configuration.")
    else:
        tokenizer_config = config.get_config(tokenizers_factory.config_path())

        if tokenizers_factory.can_build(tokenizer_config, context).type == CanBuildResultType.CAN_BUILD:
            tokenizers = tokenizers_factory.build(tokenizer_config, context)
            for key, tokenizer in tokenizers.items():
                context_key = tokenizer_config.base_path
                context_key.append(key)
                context.set(context_key, tokenizer)

            if not config.has_path(data_sources_factory.config_path()):
                print(f"Missing loader configuration.")
            else:
                data_sources_config = config.get_config(data_sources_factory.config_path())

                if data_sources_factory.can_build(data_sources_config, context):
                    data_sources = data_sources_factory.build(data_sources_config, context)
                    print(data_sources)

        else:
            print("Error")

        if config.has_path(module_factory.config_path()):

            module_config = config.get_config(module_factory.config_path())
            if module_factory.can_build(module_config, context).type == CanBuildResultType.CAN_BUILD:
                module = module_factory.build(module_config, context)


    #builder.register_handler(tokenizer_handler)

    #container = builder.build(config)

    #print(container.context)
