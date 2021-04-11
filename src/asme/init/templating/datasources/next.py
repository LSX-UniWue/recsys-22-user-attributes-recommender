from typing import Dict, Any

from asme.init.templating.datasources.datasources import build_datasource, DataSourceTemplateProcessor, \
    NextPositionDatasetBuilder, LeaveOneOutNextPositionDatasetBuilder, LeaveOneOutSessionDatasetBuilder, \
    TARGET_EXTRACTOR_PROCESSOR_CONFIG


class NextSequenceStepDataSourcesTemplateProcessor(DataSourceTemplateProcessor):

    """
     This data sources template processor configs the datasets in the following was:
    - train: a nextitem datasource with a tokenizer processor
    - validation: a nextitem datasource with a tokenizer processor and a target extractor
    - test: a nextitem datasource with a tokenizer processor and a target extractor
    """

    TRAIN_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutNextPositionDatasetBuilder()]
    TEST_VALID_DATASET_BUILDERS = [NextPositionDatasetBuilder(), LeaveOneOutSessionDatasetBuilder()]

    def _get_template_key(self) -> str:
        return 'next_sequence_step_data_sources'

    def _build_train_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.TRAIN_DATASET_BUILDERS, parser, config, 'train')

    def _build_validation_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, parser, config, 'validation')

    def _build_test_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource(self.TEST_VALID_DATASET_BUILDERS, parser, config, 'test',
                                [TARGET_EXTRACTOR_PROCESSOR_CONFIG])
