from typing import Dict, Any

from init.templating.datasources.mask import MaskDataSourcesTemplateProcessor
from init.templating.datasources.next import NextSequenceStepDataSourcesTemplateProcessor
from init.templating.datasources.positive_negative import PositiveNegativeDataSourcesTemplateProcessor
from init.templating.search.processor import SearchTemplateProcessor
from init.templating.search.resolver import OptunaParameterResolver
from init.templating.template_processor import TemplateProcessor
from init.templating.trainer.output_directory import OutputDirectoryProcessor


class TemplateEngine:
    """
    An engine the consecutively applies template processors to the configuration.
    """

    def __init__(self):
        super(TemplateEngine, self).__init__()
        self._templates = [MaskDataSourcesTemplateProcessor(),
                           PositiveNegativeDataSourcesTemplateProcessor(),
                           NextSequenceStepDataSourcesTemplateProcessor(),
                           OutputDirectoryProcessor()]

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply all template processors (in order) to the configuration.

        :param config: a config containing templates.
        :return: the updated config after applying all processors.
        """
        for template in self._templates:
            if template.can_modify(config):
                config = template.modify(config)
        return config

    def add_processor(self, processor: TemplateProcessor):
        """
        Adds a processor to the end of the list of applied processors.

        :param processor: a processor.
        :return:
        """
        self._templates.append(processor)
