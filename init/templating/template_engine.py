from typing import Dict, Any

from init.templating import TEMPLATES_CONFIG_KEY
from init.templating.datasources.mask import MaskDataSourcesTemplateProcessor
from init.templating.datasources.next import NextSequenceStepDataSourcesTemplateProcessor
from init.templating.datasources.positive_negative import PositiveNegativeDataSourcesTemplateProcessor
from init.templating.trainer.output_directory import OutputDirectoryProcessor


class TemplateEngine:

    """
    the template engine holds all templates processors and modifies the config via the configuired template processors
    """

    def __init__(self):
        super(TemplateEngine, self).__init__()
        self._template_processors = [MaskDataSourcesTemplateProcessor(),
                                     PositiveNegativeDataSourcesTemplateProcessor(),
                                     NextSequenceStepDataSourcesTemplateProcessor(),
                                     OutputDirectoryProcessor()]

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if TEMPLATES_CONFIG_KEY not in config:
            # no templates in config -> nothing to do
            return config
        for template_processors in self._template_processors:
            if template_processors.can_modify(config):
                config = template_processors.modify(config)

        # after all templates are applied remove the templates element from the config
        # check if every
        config.pop(TEMPLATES_CONFIG_KEY)
        return config
