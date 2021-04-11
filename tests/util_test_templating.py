from typing import Dict, Any

from torch.utils.data import DataLoader

from asme.init.config import Config
from asme.init.context import Context
from asme.init.factories.data_sources.data_sources import DataSourcesFactory
from asme.init.templating.template_engine import TemplateEngine
from data.datasets import SAMPLE_IDS, ITEM_SEQ_ENTRY_NAME, TARGET_ENTRY_NAME
from util_test_tokenizer import create_tokenizer


"""
the number of sequences in the example csv file
"""
NUM_EXAMPLES_SEQUENCES = 10


def load_dataset(template: Dict[str, Any]
                 ) -> Dict[str, DataLoader]:
    modified_template = TemplateEngine().modify(template)
    print(modified_template)
    context = Context()
    context.set('tokenizers.item', create_tokenizer())

    config = Config(modified_template.get('data_sources'))
    return DataSourcesFactory().build(config, context)


def get_all_data(data_loader: DataLoader
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
