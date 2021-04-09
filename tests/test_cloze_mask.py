from pytorch_lightning import seed_everything

from util_test import assert_list_equal
from util_test_tokenizer import create_tokenizer
from data.datasets import ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from data.datasets.processors.cloze_mask import ClozeMaskProcessor


def test_cloze_mask_processor_last_item():
    seed_everything(42)
    tokenizer = create_tokenizer()
    processor = ClozeMaskProcessor({"tokenizers.item": tokenizer}, 1.0, 1.0, 42)

    parsed_session = {
        ITEM_SEQ_ENTRY_NAME: [5, 8, 9, 7, 3, 4]
    }
    masked = processor.process(parsed_session)

    assert_list_equal(masked[ITEM_SEQ_ENTRY_NAME], [5, 8, 9, 7, 3, tokenizer.mask_token_id])
    expected_targets = [tokenizer.pad_token_id] * 5 + [4]
    assert_list_equal(masked[TARGET_ENTRY_NAME], expected_targets)


def test_cloze_mask_processor():
    seed_everything(42)
    tokenizer = create_tokenizer()

    processor = ClozeMaskProcessor({"tokenizers.item": tokenizer}, 0.5, 0.1, 42)

    parsed_session = {
        ITEM_SEQ_ENTRY_NAME: [5, 8, 9, 7, 3, 4, 12, 10, 11, 3]
    }
    masked = processor.process(parsed_session)

    assert_list_equal(masked[ITEM_SEQ_ENTRY_NAME], [1, 1, 1, 7, 3, 4, 1, 0, 1, 1])
    expected_targets = [5, 8, 9, 0, 0, 0, 12, 10, 11, 3]
    assert_list_equal(masked[TARGET_ENTRY_NAME], expected_targets)
