from typing import Dict, Any

from init.templating.datasources.datasources import build_datasource, DataSourceTemplateProcessor, \
    LeaveOneOutSessionDatasetBuilder, NextPositionDatasetBuilder, ConditionalSequenceOrSequencePositionDatasetBuilder


class PositiveNegativeDataSourcesTemplateProcessor(DataSourceTemplateProcessor):

    """
     This data sources template processor configs the datasets in the following was:
    - train: a session datasource with a tokenizer and a positive negative sampler processor
    - validation: a nextitem datasource with a tokenizer processor
    - test: a nextitem datasource with a tokenizer processor
    """

    TRAIN_DATASET_BUILDERS = [ConditionalSequenceOrSequencePositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def _get_template_key(self) -> str:
        return 'pos_neg_data_sources'

    def _build_train_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        seed = config['seed']
        pos_neg_sampler_processor = {
            'type': "pos_neg",
            'seed': seed
        }

        return build_datasource(self.TRAIN_DATASET_BUILDERS, parser, config, 'train', pos_neg_sampler_processor)

    def _build_validation_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, parser, config, 'validation')

    def _build_test_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, parser, config, 'test')
