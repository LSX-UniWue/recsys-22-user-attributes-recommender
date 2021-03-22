import json
from pathlib import Path
from typing import List

import _jsonnet

from asme.init.config import Config
from asme.init.container import Container
from asme.init.context import Context
from asme.init.factories.container import ContainerFactory
from asme.init.templating.template_engine import TemplateEngine
from asme.init.templating.template_processor import TemplateProcessor

""" key to retrieve the object metric used in hyperparameter study """
OBJECTIVE_METRIC_KEY = 'objective_metric'


def load_config(config_file: Path,
                additional_head_processors: List[TemplateProcessor] = [],
                additional_tail_processors: List[TemplateProcessor] = []
                ) -> Config:
    config_file = Path(config_file)

    if not config_file.exists():
        print(f"the config file cannot be found. Please check the path '{config_file}'!")
        exit(-1)

    config_json = _jsonnet.evaluate_file(str(config_file))

    loaded_config = json.loads(config_json)

    template_engine = TemplateEngine(head_processors=additional_head_processors,
                                     tail_processors=additional_tail_processors)

    config_to_use = template_engine.modify(loaded_config)
    return Config(config_to_use)


def create_container(config: Config) -> Container:
    context = Context()

    container_factory = ContainerFactory()
    container = container_factory.build(config, context)

    return container


def load_container(config_file: Path) -> Container:
    config_raw = load_config(config_file)
    return create_container(config_raw)
