from typing import Dict, Any

from init.templating.datasources.mask import MaskDataSourcesTemplateProcessor
from init.templating.datasources.next import NextSequenceStepDataSourcesTemplateProcessor
from init.templating.datasources.positive_negative import PositiveNegativeDataSourcesTemplateProcessor
from init.templating.trainer.output_directory import OutputDirectoryProcessor


class TemplateEngine:

    def __init__(self):
        super(TemplateEngine, self).__init__()
        self._templates = [MaskDataSourcesTemplateProcessor(),
                           PositiveNegativeDataSourcesTemplateProcessor(),
                           NextSequenceStepDataSourcesTemplateProcessor(),
                           OutputDirectoryProcessor()]

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        for template in self._templates:
            if template.can_modify(config):
                config = template.modify(config)
        return config
