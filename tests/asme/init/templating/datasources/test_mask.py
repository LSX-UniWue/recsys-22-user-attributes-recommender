from typing import Any, Dict, List

from pathlib import Path

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, RandomSampler

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.data_sources.data_sources import DataSourcesFactory
from asme.init.templating.template_engine import TemplateEngine
from asme.tokenization.tokenizer import Tokenizer
from asme.tokenization.vocabulary import CSVVocabularyReaderWriter
from data.datasets import SAMPLE_IDS, ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME

TEST_DATASET_BASE_PATH = Path('../../../../example_dataset/')

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


def _create_tokenizer() -> Tokenizer:
    with open(TEST_DATASET_BASE_PATH / 'example.vocabulary.item_id.txt') as vocab_file:
        vocab_reader = CSVVocabularyReaderWriter()
        vocab = vocab_reader.read(vocab_file)
        return Tokenizer(vocab, pad_token='<PAD>', mask_token='<MASK>', unk_token='<UNK>')


def _get_all_data(data_loader: DataLoader) -> Dict[int, Any]:
    data = {}
    for batch in data_loader:
        data_ids = batch[SAMPLE_IDS].tolist()
        sequences = batch[ITEM_SEQ_ENTRY_NAME].tolist()
        targets = batch[TARGET_ENTRY_NAME].tolist()

        for data_id, sequence, target in zip(data_ids, sequences, targets):
            data[data_id] = sequence, target

    return data


def _assert_list_equal(list1: List[Any],
                       list2: List[Any]) -> None:
    assert len(list1) == len(list2)
    assert all([a == b for a, b in zip(list1, list2)])


def test_leave_one_out_mask_template():
    seed_everything(42)
    modified_template = TemplateEngine().modify(LEAVE_ONE_OUT_TEMPLATE)
    context = Context()
    context.set('tokenizers.item', _create_tokenizer())

    config = Config(modified_template.get('data_sources'))
    data_sources = DataSourcesFactory().build(config, context)

    train_dataloader = data_sources['train']

    # check if train data loader is shuffled
    assert isinstance(train_dataloader.sampler, RandomSampler)

    train_data = _get_all_data(train_dataloader)
    assert len(train_data) == 5

    # test some train data
    sequence, target = train_data[0]
    _assert_list_equal(sequence, [3, 1, 0, 0])
    _assert_list_equal(target, [0, 4, 0, 0])
    seq2, target2 = train_data[4]
    _assert_list_equal(seq2, [1, 0, 0, 0])
    _assert_list_equal(target2, [7, 0, 0, 0])

    # test some test data
    test_dataloader = data_sources['test']
    test_data = _get_all_data(test_dataloader)

    test_sequence, test_target = test_data[2]
    _assert_list_equal(test_sequence, [9, 10, 11, 1])
    assert test_target == 12

    # test some validation data
    val_dataloader = data_sources['validation']
    val_data = _get_all_data(val_dataloader)

    val_sequence, val_target = val_data[1]
    _assert_list_equal(val_sequence, [3, 4, 1, 0])
    assert val_target == 7
