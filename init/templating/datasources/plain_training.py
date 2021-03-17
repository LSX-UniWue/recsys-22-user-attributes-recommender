from typing import Dict, Any

from init.templating.datasources.datasources import DataSourceTemplateProcessor, build_datasource, \
    NextPositionDatasetBuilder, LeaveOneOutNextPositionDatasetBuilder, SequenceDatasetRatioSplitBuilder


class PlainTrainingSourcesTemplateProcessor(DataSourceTemplateProcessor):

    """
     This data sources template processor configs the datasets in the following was:
    - train: a item sequence datasource with a tokenizer processor
    - validation: a nextitem datasource with a tokenizer processor
    - test: a nextitem datasource with a tokenizer processor
    """

    DATASET_BUILDER_TRAINING = [SequenceDatasetRatioSplitBuilder()]
    DATASET_BUILDERS_VALIDATION_AND_TEST = [NextPositionDatasetBuilder()]

    def _get_template_key(self) -> str:
        return 'plain_training_next_item_test_and_validation_data_sources'

    def _build_train_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.DATASET_BUILDER_TRAINING, parser, config, 'train')

    def _build_validation_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.DATASET_BUILDERS_VALIDATION_AND_TEST, parser, config, 'validation')

    def _build_test_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.DATASET_BUILDERS_VALIDATION_AND_TEST, parser, config, 'test')
