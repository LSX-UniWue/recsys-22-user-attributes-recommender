from typing import Dict, Any

from init.templating.datasources.mask import build_datasource, build_parser_config
from init.templating.template_processor import TemplateProcessor


class PositiveNegativeDataSourcesTemplateProcessor(TemplateProcessor):

    # FIXME: duplicate code
    def can_modify(self, config: Dict[str, Any]) -> bool:
        template_present = "pos_neg_data_sources" in config
        if "data_sources" in config and template_present:
            raise KeyError('data_sources already specified. Cannot create a templating')

        return template_present

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        data = config.pop('pos_neg_data_sources')

        parser_config = data['parser']
        parser = build_parser_config(parser_config)

        seed = data['seed']
        pos_neg_sampler_processor = {
            'type': "pos_neg",
            'seed': seed
        }

        train_config = build_datasource("session", parser, data, 'train', pos_neg_sampler_processor)
        validation_config = build_datasource("nextit", parser, data, 'validation')
        test_config = build_datasource("nextit", parser, data, 'test')

        config['data_sources'] = {
            'train': train_config,
            'test': test_config,
            'validation': validation_config
        }

        return config