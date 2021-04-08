from typing import Any, Dict

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, RandomSampler

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.data_sources.data_sources import DataSourcesFactory
from asme.init.templating.template_engine import TemplateEngine
from util_test import assert_list_equal
from util_test_tokenizer import TEST_DATASET_BASE_PATH, create_tokenizer
from data.datasets import SAMPLE_IDS, ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME


LEAVE_ONE_OUT_TEMPLATE = {
        "templates": {
            'mask_data_sources': {
                'parser': {
                    'item_column_name': "item_id"
                },
                'loader': {
                    'batch_size': 2,
                    'max_seq_length': 4,
                    'num_workers': 0
                },
                'path': TEST_DATASET_BASE_PATH,
                'file_prefix': 'example',
                'mask_probability': 0.1,
                'mask_seed': 123456,
                'split_type': 'leave_one_out'
            }
        }
    }

RATIO_TEMPLATE = {
    "templates": {
        'mask_data_sources': {
            'parser': {
                'item_column_name': "item_id"
            },
            'loader': {
                'batch_size': 2,
                'max_seq_length': 4,
                'num_workers': 0
            },
            'path': TEST_DATASET_BASE_PATH / 'ratio-0.8_0.1_0.1',
            'file_prefix': 'example',
            'mask_probability': 0.1,
            'mask_seed': 123456
        }
    }
}


def _get_all_data(data_loader: DataLoader
                  ) -> Dict[str, Any]:
    data = {}
    for batch in data_loader:
        data_ids = batch[SAMPLE_IDS].tolist()
        positions = batch['pos'].tolist() if 'pos' in batch else [0] * len(data_ids)
        sequences = batch[ITEM_SEQ_ENTRY_NAME].tolist()
        targets = batch[TARGET_ENTRY_NAME].tolist()

        for data_id, pos, sequence, target in zip(data_ids, positions, sequences, targets):
            data[f'{data_id}_{pos}'] = sequence, target

    return data


def _load_dataset(template: Dict[str, Any]
                  ) -> Dict[str, DataLoader]:
    modified_template = TemplateEngine().modify(template)
    context = Context()
    context.set('tokenizers.item', create_tokenizer())

    config = Config(modified_template.get('data_sources'))
    return DataSourcesFactory().build(config, context)


def test_leave_one_out_mask_template():
    seed_everything(42)

    data_sources = _load_dataset(LEAVE_ONE_OUT_TEMPLATE)
    train_dataloader = data_sources['train']

    # check if train data loader is shuffled
    assert isinstance(train_dataloader.sampler, RandomSampler)

    train_data = _get_all_data(train_dataloader)
    assert len(train_data) == 10

    # test some train data
    sequence, target = train_data['0_2']
    assert_list_equal(sequence, [1, 4, 0, 0])
    assert_list_equal(target, [3, 0, 0, 0])
    seq2, target2 = train_data['4_1']
    assert_list_equal(seq2, [7, 0, 0, 0])
    assert_list_equal(target2, [0, 0, 0, 0])

    # test some test data
    test_dataloader = data_sources['test']
    test_data = _get_all_data(test_dataloader)

    test_sequence, test_target = test_data['2_3']
    assert_list_equal(test_sequence, [9, 10, 11, 1])
    assert test_target == 12

    # test some validation data
    val_dataloader = data_sources['validation']
    val_data = _get_all_data(val_dataloader)

    val_sequence, val_target = val_data['1_2']
    assert_list_equal(val_sequence, [3, 4, 1, 0])
    assert val_target == 7


def test_ratio_mask_template():
    seed_everything(42)
    data_sources = _load_dataset(RATIO_TEMPLATE)

    train_dataloader = data_sources['train']
    assert isinstance(train_dataloader.sampler, RandomSampler)

    train_data = _get_all_data(train_dataloader)
    assert len(train_data) == 8

    sequence, target = train_data['0_0']
    assert_list_equal(sequence, [3, 10, 6, 0])
    assert_list_equal(target, [0, 0, 0, 0])

    seq2, target2 = train_data['4_0']
    assert_list_equal(seq2, [7, 8, 1, 0])
    assert_list_equal(target2, [0, 0, 3, 0])

    test_dataloader = data_sources['test']
    test_data = _get_all_data(test_dataloader)

    assert len(test_data) == 2
    test_seq, test_target = test_data['0_1']
    assert_list_equal(test_seq, [5, 1, 0, 0])
    assert test_target == 6

    test_seq2, test_target2 = test_data['0_2']
    assert_list_equal(test_seq2, [5, 6, 1, 0])
    assert test_target2 == 8

    val_dataloader = data_sources['validation']
    val_data = _get_all_data(val_dataloader)

    assert len(val_data) == 3

    val_seq, val_target = val_data['0_1']
    assert_list_equal(val_seq, [3, 1, 0, 0])
    assert val_target == 4

    val_seq2, val_target2 = val_data['0_3']
    assert_list_equal(val_seq2, [3, 4, 7, 1])
    assert val_target2 == 8
