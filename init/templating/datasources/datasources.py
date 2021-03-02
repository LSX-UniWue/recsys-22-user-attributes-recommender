from abc import abstractmethod
from typing import Dict, Any, Optional

from init.templating import TEMPLATES_CONFIG_KEY
from init.templating.template_processor import TemplateProcessor


CONFIG_DATASOURCES_KEY = 'data_sources'


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


def build_datasource(datasource_type: str,
                     split_type: Optional[str],
                     parser: Dict[str, Any],
                     config: Dict[str, Any],
                     prefix_id: str,
                     processor: Dict[str, Any] = None
                     ) -> Dict[str, Any]:
    """
    builds a datasource config with the specified parser, processor,
    :param datasource_type: type of data source, can be either `nextit` or `session`.
    :param split_type: type of split used, can be either `ratio` or `loo`, only necessary iff datasource_type == 'nextit'.
    :param parser:
    :param config:
    :param prefix_id:
    :param processor:
    :return:
    """
    base_path = config['path']

    loader_config = config['loader']

    base_batch_size = loader_config.get('batch_size', 0)
    batch_size = loader_config.get(f'{prefix_id}_batch_size', base_batch_size)
    max_seq_length = loader_config['max_seq_length']

    prefix = config.get(f'{prefix_id}_file_prefix', prefix_id)

    processors = [
        {
            'type': 'tokenizer'
        }
    ]

    if processor is not None:
        processors.append(processor)

    dataset_config = {
        'type': datasource_type,
        'csv_file': f'{base_path}{prefix}.csv',
        'csv_file_index': f'{base_path}{prefix}.session.idx',
        'parser': parser,
        'processors': processors
    }

    if "nextit" == datasource_type:
        next_prefix = config.get(f'{prefix_id}_index_file_prefix', prefix)
        if split_type is None:
            raise KeyError("Split type needs to be specified for `nextit` type data sources.")
        if split_type == "ratio":
            dataset_config['nip_index_file'] = f'{base_path}{next_prefix}.{prefix_id}.nextitem.idx'
        elif split_type == "loo":
            dataset_config['nip_index_file'] = f'{base_path}{next_prefix}.{prefix_id}.loo.idx'
        else:
            raise KeyError("Unknown split type.")

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
