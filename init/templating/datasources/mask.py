from typing import Dict, Any

from init.templating.template_processor import TemplateProcessor


class MaskDataSourcesTemplateProcessor(TemplateProcessor):

    def can_modify(self, config: Dict[str, Any]) -> bool:
        template_present = "mask_data_sources" in config
        if "data_sources" in config and template_present:
            raise KeyError('data_sources already specified. Cannot create a templating')

        return template_present

    def modify(self, config: Dict[str, Any]) -> Dict[str, Any]:
        data = config.pop('mask_data_sources')

        parser_config = data['parser']
        parser = build_parser_config(parser_config)

        mask_probability = data.get('mask_probability', 0.2)
        only_last_item_mask_prob = data.get('only_last_item_mask_prob', 0.1)
        seed = data['seed']
        cloze_processor = {
            'type': "cloze",
            'mask_probability': mask_probability,
            'only_last_item_mask_prob': only_last_item_mask_prob,
            'seed': seed
        }

        last_item_mask_processor = {
            'type': 'last_item_mask'
        }

        train_config = build_datasource("nextit", parser, data, 'train', cloze_processor)
        validation_config = build_datasource("nextit", parser, data, 'validation', last_item_mask_processor)
        test_config = build_datasource("nextit", parser, data, 'test', last_item_mask_processor)

        config['data_sources'] = {
            'train': train_config,
            'test': test_config,
            'validation': validation_config
        }
        return config

# FIXME: document
def build_datasource(datasource_type: str,
                     parser: Dict[str, Any],
                     config: Dict[str, Any],
                     prefix_id: str,
                     processor: Dict[str, Any] = None
                     ) -> Dict[str, Any]:
    base_path = config['path']

    base_batch_size = config.get('batch_size', 0)
    batch_size = config.get(f'{prefix_id}_batch_size', base_batch_size)
    max_seq_length = config['max_seq_length']

    prefix = config.get(f'{prefix_id}_file_prefix', prefix_id)

    processors = [
        {
            'type': 'tokenizer'
        }
    ]

    if processor is not None:
        processors.append(processor)

    return {
        'loader': {
            'dataset': {
                'type': datasource_type,
                'csv_file': f'{base_path}{prefix}.csv',
                'csv_file_index': f'{base_path}{prefix}.idx',
                'nip_index_file': f'{base_path}{prefix}.nip.idx',
                'parser': parser,
                'processors': processors
            },
            'batch_size': batch_size,
            'max_seq_length': max_seq_length
        }
    }


def build_parser_config(parser_config: Dict[str, Any]) -> Dict[str, Any]:
    item_column_name = parser_config['item_column_name']

    parser = {
        'item_column_name': item_column_name
    }

    max_seq_step_length = parser_config.get('max_seq_step_length', None)
    if max_seq_step_length is not None:
        parser['max_seq_step_length'] = max_seq_step_length

    return parser