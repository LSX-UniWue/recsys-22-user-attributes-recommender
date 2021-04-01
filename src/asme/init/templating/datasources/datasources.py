from abc import abstractmethod
from enum import Enum
from typing import Dict, Any, List

from asme.init.templating import TEMPLATES_CONFIG_KEY
from asme.init.templating.template_processor import TemplateProcessor


CONFIG_DATASOURCES_KEY = 'data_sources'


class DatasetSplit(Enum):

    """
    all supported dataset splits
    """

    RATIO_SPLIT = 1
    LEAVE_ONE_OUT = 2


class DataSourceTemplateProcessor(TemplateProcessor):

    """
    An abstract template processor that builds a data_sources definition (train, validation, test dataset)
    """

    def can_modify(self, config: Dict[str, Any]) -> bool:
        if TEMPLATES_CONFIG_KEY not in config:
            return False

        template_config = config.get(TEMPLATES_CONFIG_KEY)
        template_present = self._get_template_key() in template_config
        if CONFIG_DATASOURCES_KEY in template_config and template_present:
            raise KeyError('data_sources already specified. Can not apply template.')

        return template_present

    @abstractmethod
    def _get_template_key(self) -> str:
        pass

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        template_config = config.get(TEMPLATES_CONFIG_KEY)
        data = template_config.pop(self._get_template_key())

        parser_config = data['parser']
        parser = build_parser_config(parser_config)

        train_config = self._build_train_datasource(data, parser)
        validation_config = self._build_validation_datasource(data, parser)
        test_config = self._build_test_datasource(data, parser)

        config[CONFIG_DATASOURCES_KEY] = {
            'train': train_config,
            'test': test_config,
            'validation': validation_config
        }
        return config

    @abstractmethod
    def _build_train_datasource(self,
                                config: Dict[str, Any],
                                parser: Dict[str, Any]
                                ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _build_validation_datasource(self,
                                     config: Dict[str, Any],
                                     parser: Dict[str, Any]
                                     ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _build_test_datasource(self,
                               config: Dict[str, Any],
                               parser: Dict[str, Any]
                               ) -> Dict[str, Any]:
        pass


PARSER_ITEM_COLUMN_NAME = 'item_column_name'
PARSER_ITEM_SEPARATOR = 'item_separator'


def build_parser_config(parser_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    builds the parser config, currently just pass the config, because nothing is currently modified by the datasources
    templates
    :param parser_config:
    :return:
    """
    return parser_config


def _get_prefix(config: Dict[str, Any],
                prefix_id: str
                ) -> str:
    prefix = config.get('file_prefix', prefix_id)
    prefix = config.get(f'{prefix_id}_file_prefix', prefix)
    return prefix


class DatasetBuilder:

    @abstractmethod
    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit) -> bool:
        pass

    @abstractmethod
    def build_dataset_definition(self, prefix_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        pass


class NextPositionDatasetBuilder(DatasetBuilder):
    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit) -> bool:
        return dataset_split_type == DatasetSplit.RATIO_SPLIT

    def build_dataset_definition(self, prefix_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        base_path = config['path']
        prefix = _get_prefix(config, prefix_id)
        next_seq_step_type = config.get('next_seq_step_type', 'nextitem')
        return {
            'type': 'sequence_position',
            'csv_file': f'{base_path}{prefix}.{prefix_id}.csv',
            'csv_file_index': f'{base_path}{prefix}.{prefix_id}.session.idx',
            'nip_index_file': f'{base_path}{prefix}.{prefix_id}.{next_seq_step_type}.idx'
        }


class SequenceDatasetRatioSplitBuilder(DatasetBuilder):

    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit) -> bool:
        return dataset_split_type == DatasetSplit.RATIO_SPLIT

    def build_dataset_definition(self, prefix_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        base_path = config['path']
        prefix = _get_prefix(config, prefix_id)
        return {
            'type': 'session',
            'csv_file': f'{base_path}{prefix}.{prefix_id}.csv',
            'csv_file_index': f'{base_path}{prefix}.{prefix_id}.session.idx'
        }


class ConditionalSequenceOrSequencePositionDatasetBuilder(DatasetBuilder):

    def __init__(self):
        super().__init__()
        self._sequence_dataset_builder = SequenceDatasetRatioSplitBuilder()
        self._sequence_position_dataset_builder = NextPositionDatasetBuilder()

    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit) -> bool:
        return dataset_split_type == DatasetSplit.RATIO_SPLIT

    def build_dataset_definition(self, prefix_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        if 'next_seq_step_type' in config:
            dataset_config = self._sequence_position_dataset_builder.build_dataset_definition(prefix_id, config)
            dataset_config['add_target'] = False
            return dataset_config

        return self._sequence_dataset_builder.build_dataset_definition(prefix_id, config)


class LeaveOneOutNextPositionDatasetBuilder(DatasetBuilder):

    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit) -> bool:
        return dataset_split_type == DatasetSplit.LEAVE_ONE_OUT

    def build_dataset_definition(self, prefix_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        base_path = config['path']
        prefix = _get_prefix(config, prefix_id)

        #FIXME (AD) we need to reiterate our  dataset schema for loo
        if prefix_id == "train":
            # (AD) we should rename the index file to conform to validation/test
            dataset_config = {
                'type': 'sequence_position',
                'csv_file': f'{base_path}{prefix}.csv',
                'csv_file_index': f'{base_path}{prefix}.session.idx',
                'nip_index_file': f'{base_path}/loo/{prefix}.{prefix_id}.nextitem.idx'
            }
        else:
            dataset_config = {
                'type': 'sequence_position',
                'csv_file': f'{base_path}{prefix}.csv',
                'csv_file_index': f'{base_path}{prefix}.session.idx',
                'nip_index_file': f'{base_path}/loo/{prefix}.{prefix_id}.loo.idx'
            }

        return dataset_config


class LeaveOneOutSessionDatasetBuilder(DatasetBuilder):

    def can_build_dataset_definition(self, dataset_split_type: DatasetSplit):
        return dataset_split_type == DatasetSplit.LEAVE_ONE_OUT

    def build_dataset_definition(self, prefix_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        base_path = config['path']
        prefix = _get_prefix(config, prefix_id)
        index_file_path = f"{prefix}.{prefix_id}"
        is_training = prefix_id == 'train'
        if is_training:
            validation_prefix = _get_prefix(config, 'validation')
            index_file_path = f"{validation_prefix}.validation"
        dataset_config = {
            'type': 'sequence_position',
            'csv_file': f'{base_path}{prefix}.csv',
            'csv_file_index': f'{base_path}{prefix}.session.idx',
            'nip_index_file': f'{base_path}loo/{index_file_path}.loo.idx'
        }
        if is_training:
            dataset_config['add_target'] = False
        return dataset_config


def build_datasource(dataset_builders: List[DatasetBuilder],
                     parser: Dict[str, Any],
                     config: Dict[str, Any],
                     prefix_id: str,
                     processor: Dict[str, Any] = None
                     ) -> Dict[str, Any]:
    """
    builds a datasource config with the specified parser, processor,
    :param dataset_builders: the builders to use to build the dataset config
    :param parser:
    :param config:
    :param prefix_id:
    :param processor:
    :return:
    """
    loader_config = config['loader']
    base_batch_size = loader_config.get('batch_size', 0)
    batch_size = loader_config.get(f'{prefix_id}_batch_size', base_batch_size)
    max_seq_length = loader_config['max_seq_length']

    processors = [
        {
            'type': 'tokenizer'
        }
    ]

    if processor is not None:
        processors.append(processor)

    dataset_split_type = DatasetSplit[config.get('split_type', DatasetSplit.RATIO_SPLIT.name).upper()]

    def _build_dataset_config() -> Dict[str, Any]:
        for datasource_builder in dataset_builders:
            if datasource_builder.can_build_dataset_definition(dataset_split_type):
                return datasource_builder.build_dataset_definition(prefix_id, config)
        raise ValueError('no datasource builder found')

    dataset_config = _build_dataset_config()
    dataset_config['parser'] = parser
    dataset_config['processors'] = processors

    loader_config_dict = {
        'dataset': dataset_config,
        'batch_size': batch_size,
        'max_seq_length': max_seq_length
    }

    max_seq_step_length = loader_config.get('max_seq_step_length')
    if max_seq_step_length is not None:
        loader_config_dict['max_seq_step_length'] = max_seq_step_length

    num_workers = loader_config.get('num_workers')
    if num_workers is not None:
        loader_config_dict['num_workers'] = num_workers

    return {
        'loader': loader_config_dict
    }
