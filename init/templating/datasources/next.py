from typing import Dict, Any

from init.templating.datasources.datasources import build_datasource, DataSourceTemplateProcessor


class NextSequenceStepDataSourcesTemplateProcessor(DataSourceTemplateProcessor):

    """
     This data sources template processor configs the datasets in the following was:
    - train: a nextitem datasource with a tokenizer processor
    - validation: a nextitem datasource with a tokenizer processor
    - test: a nextitem datasource with a tokenizer processor
    """

    def _get_template_key(self) -> str:
        return 'next_sequence_step_data_sources'

    def _build_train_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource("nextit", parser, config, 'train')

    def _build_validation_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource("nextit", parser, config, 'validation')

    def _build_test_datasource(self, config: Dict[str, Any], parser: Dict[str, Any]) -> Dict[str, Any]:
        return build_datasource("nextit", parser, config, 'test')
