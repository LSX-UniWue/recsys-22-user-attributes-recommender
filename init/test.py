from init.config import Config
from init.config_builder import ContainerBuilder
from init.context import Context
from init.factories.tokenizer_factory import TokenizersFactory
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

    tokenizer_factory = TokenizersFactory()

    config = Config(config)
    context = Context()

    if not config.has_path(tokenizer_factory.config_path()):
        print(f"Missing tokenizer configuration.")
    else:
        tokenizer_config = config.get_config(tokenizer_factory.config_path())

        if tokenizer_factory.can_build(tokenizer_config, context).type == CanBuildResultType.CAN_BUILD:
            tokenizer = tokenizer_factory.build(tokenizer_config, context)
            print(tokenizer)
        else:
            print("Error")


    #builder.register_handler(tokenizer_handler)

    #container = builder.build(config)

    #print(container.context)