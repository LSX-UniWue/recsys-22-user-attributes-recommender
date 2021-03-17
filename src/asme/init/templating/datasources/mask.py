from typing import Dict, Any

from data.datasets import ITEM_SEQ_ENTRY_NAME

from asme.init.templating.datasources.datasources import build_datasource, DataSourceTemplateProcessor, \
    SequenceDatasetRatioSplitBuilder, LeaveOneOutSessionDatasetBuilder, NextPositionDatasetBuilder


class MaskDataSourcesTemplateProcessor(DataSourceTemplateProcessor):

    """
    This data sources template processor configs the datasets in the following was:
    - train: a nextitem datasource with a tokenizer and cloze processor
    - validation: a nextitem datasource with a tokenizer and a list item mask processor
    - test: a nextitem datasource with a tokenizer and a list item mask processor
    """

    TRAIN_DATASET_BUILDERS = [SequenceDatasetRatioSplitBuilder(), LeaveOneOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def _get_template_key(self) -> str:
        return 'mask_data_sources'

    def _build_train_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        mask_probability = config.get('mask_probability', 0.2)
        only_last_item_mask_prob = config.get('only_last_item_mask_prob', 0.1)
        seed = config['mask_seed']
        masking_targets = config.get('mask_additional_attributes', []) + [ITEM_SEQ_ENTRY_NAME]
        cloze_processor = {
            'type': "cloze",
            'mask_probability': mask_probability,
            'only_last_item_mask_prob': only_last_item_mask_prob,
            'seed': seed,
            'masking_targets': masking_targets
        }

        return build_datasource(self.TRAIN_DATASET_BUILDERS, parser, config, 'train', cloze_processor)

    def _build_validation_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        masking_targets = config.get('mask_additional_attributes', []) + [ITEM_SEQ_ENTRY_NAME]
        mask_last_item_processor = {
            'type': 'last_item_mask',
            'masking_targets': masking_targets
        }
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, parser, config, 'validation', mask_last_item_processor)

    def _build_test_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        masking_targets = config.get('mask_additional_attributes', []) + [ITEM_SEQ_ENTRY_NAME]
        mask_last_item_processor = {
            'type': 'last_item_mask',
            'masking_targets': masking_targets
        }
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, parser, config, 'test', mask_last_item_processor)
