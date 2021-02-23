from init.config import Config
from init.context import Context
from init.factories.container import ContainerFactory

if __name__ == "__main__":
    import json
    import _jsonnet

    from pathlib import Path

    config_file = Path("../projects/examples/bert4rec_config.jsonnet")
    config_json = _jsonnet.evaluate_file(str(config_file))

    config_raw = json.loads(config_json)

    config = Config(config_raw)
    context = Context()

    container_factory = ContainerFactory()

    container = container_factory.build(config, context)

    container.trainer().build().fit(container.module(), container.train_dataloader(), val_dataloaders=container.validation_dataloader())

