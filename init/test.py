from init.config import Config
from init.context import Context
from init.factories.container import ContainerFactory
from init.factories.data_sources.data_sources import DataSourcesFactory
from init.factories.modules.modules import ModuleFactory
from init.factories.tokenizer.tokenizer_factory import TokenizersFactory
from init.object_factory import CanBuildResultType
from runner.util.builder import TrainerBuilder

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

    container.trainer().fit(container.module(), container.train_dataloader())

