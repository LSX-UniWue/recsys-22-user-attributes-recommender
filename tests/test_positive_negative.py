from typing import Any, Dict

import pytest
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, RandomSampler

from util_test import assert_list_equal
from util_test_templating import load_dataset, get_all_data
from util_test_tokenizer import TEST_DATASET_BASE_PATH
from data.datasets import SAMPLE_IDS, ITEM_SEQ_ENTRY_NAME, POSITIVE_SAMPLES_ENTRY_NAME, NEGATIVE_SAMPLES_ENTRY_NAME


LEAVE_ONE_OUT_TEMPLATE = {
    "templates": {
        'pos_neg_data_sources': {
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
            'split_type': 'leave_one_out',
            'seed': 42  # FIXME: remove
        }
    }
}

RATIO_TEMPLATE = {
    "templates": {
        'pos_neg_data_sources': {
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
            'seed': 42  # FIXME: remove
        }
    }
}


def _get_all_data_positive_negative(data_loader: DataLoader
                                    ) -> Dict[str, Any]:
    data = {}
    for batch in data_loader:
        data_ids = batch[SAMPLE_IDS].tolist()
        positions = batch['pos'].tolist() if 'pos' in batch else [0] * len(data_ids)
        sequences = batch[ITEM_SEQ_ENTRY_NAME].tolist()
        positive_examples = batch[POSITIVE_SAMPLES_ENTRY_NAME].tolist()
        negative_examples = batch[NEGATIVE_SAMPLES_ENTRY_NAME].tolist()

        for data_id, pos, sequence, positive_example, negative_example in zip(data_ids, positions, sequences,
                                                                              positive_examples, negative_examples):
            data[f'{data_id}_{pos}'] = sequence, positive_example, negative_example

    return data


@pytest.mark.skip(reason="must be discussed with alex")
def test_leave_one_out_positive_negative_template():
    seed_everything(42)

    data_sources = load_dataset(LEAVE_ONE_OUT_TEMPLATE)
    train_dataloader = data_sources['train']

    # check if train data loader is shuffled
    assert isinstance(train_dataloader.sampler, RandomSampler)

    train_data = _get_all_data_positive_negative(train_dataloader)
    assert len(train_data) == 10

    # test some train data
    sequence, pos, neg = train_data['0_2']
    assert_list_equal(sequence, [1, 4, 0, 0])
    assert_list_equal(pos, [3, 0, 0, 0])
    assert_list_equal(neg, [3, 0, 0, 0])
    seq2, pos2, neg2 = train_data['4_1']
    assert_list_equal(seq2, [7, 0, 0, 0])
    assert_list_equal(pos2, [0, 0, 0, 0])
    assert_list_equal(neg2, [0, 0, 0, 0])

    # test some test data
    test_dataloader = data_sources['test']
    test_data = _get_all_data_positive_negative(test_dataloader)

    test_sequence, test_target = test_data['2_3']
    assert_list_equal(test_sequence, [9, 10, 11, 1])
    assert test_target == 12

    # test some validation data
    val_dataloader = data_sources['validation']
    val_data = _get_all_data_positive_negative(val_dataloader)

    val_sequence, val_target = val_data['1_2']
    assert_list_equal(val_sequence, [3, 4, 1, 0])
    assert val_target == 7


def test_ratio_positive_negative_template():
    seed_everything(42)
    data_sources = load_dataset(RATIO_TEMPLATE)

    train_dataloader = data_sources['train']
    assert isinstance(train_dataloader.sampler, RandomSampler)

    train_data = _get_all_data_positive_negative(train_dataloader)
    assert len(train_data) == 8

    sequence, pos, neg = train_data['0_0']
    assert_list_equal(sequence, [3, 10, 0, 0])
    assert_list_equal(pos, [10, 6, 0, 0])
    assert_list_equal(neg, [4, 9, 0, 0])

    seq2, pos2, neg2 = train_data['4_0']
    assert_list_equal(seq2, [7, 8, 0, 0])
    assert_list_equal(pos2, [8, 3, 0, 0])
    assert_list_equal(neg2, [9, 10, 0, 0])

    test_dataloader = data_sources['test']
    test_data = get_all_data(test_dataloader)

    assert len(test_data) == 2
    test_seq, test_target = test_data['0_1']
    assert_list_equal(test_seq, [5, 0, 0, 0])
    assert test_target == 6

    test_seq2, test_target2 = test_data['0_2']
    assert_list_equal(test_seq2, [5, 6, 0, 0])
    assert test_target2 == 8

    val_dataloader = data_sources['validation']
    val_data = get_all_data(val_dataloader)

    assert len(val_data) == 3

    val_seq, val_target = val_data['0_1']
    assert_list_equal(val_seq, [3, 0, 0, 0])
    assert val_target == 4

    val_seq2, val_target2 = val_data['0_3']
    assert_list_equal(val_seq2, [3, 4, 7, 0])
    assert val_target2 == 8
