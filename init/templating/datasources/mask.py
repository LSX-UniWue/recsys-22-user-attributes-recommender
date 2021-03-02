from typing import Dict, Any

from init.templating.datasources.datasources import build_datasource, DataSourceTemplateProcessor


class MaskDataSourcesTemplateProcessor(DataSourceTemplateProcessor):

    """
    This data sources template processor configs the datasets in the following was:
    - train: a nextitem datasource with a tokenizer and cloze processor
    - validation: a nextitem datasource with a tokenizer and a list item mask processor
    - test: a nextitem datasource with a tokenizer and a list item mask processor
    """

    LAST_ITEM_MASK_PROCESSOR_CONFIG = {
        'type': 'last_item_mask'
    }

    def _get_template_key(self) -> str:
        return 'mask_data_sources'

    def _build_train_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        mask_probability = config.get('mask_probability', 0.2)
        only_last_item_mask_prob = config.get('only_last_item_mask_prob', 0.1)
        seed = config['mask_seed']
        cloze_processor = {
            'type': "cloze",
            'mask_probability': mask_probability,
            'only_last_item_mask_prob': only_last_item_mask_prob,
            'seed': seed
        }

        return build_datasource("nextit", "loo", parser, config, 'train', cloze_processor)

    def _build_validation_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource("nextit", "loo", parser, config, 'validation', self.LAST_ITEM_MASK_PROCESSOR_CONFIG)

    def _build_test_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource("nextit", "loo", parser, config, 'test', self.LAST_ITEM_MASK_PROCESSOR_CONFIG)
