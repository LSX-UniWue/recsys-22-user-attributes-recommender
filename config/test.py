from config.config_builder import ContainerBuilder
from config.factories.tokenizer_factory import TokenizerFactory

if __name__ == "__main__":
    import json
    import _jsonnet

    from pathlib import Path

    config_file = Path("../projects/examples/bert4rec_config.jsonnet")
    config_json = _jsonnet.evaluate_file(str(config_file))

    config = json.loads(config_json)
    print(config)

    builder = ContainerBuilder()

    tokenizer_handler = TokenizerFactory()
    builder.register_handler(tokenizer_handler)

    container = builder.build(config)

    print(container.context)