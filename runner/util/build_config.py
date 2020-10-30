from typing import Dict, Any

import yaml
import copy

from configs.models.bert4rec.bert4rec_config import BERT4RecConfig

# configs from the paper
BERT4REC_MODEL_CONFIG_MOVIELENS_1M = BERT4RecConfig(item_voc_size=-1,  # set to the correct vocab size
                                                    max_seq_length=200,
                                                    num_transformer_heads=2,
                                                    num_transformer_layers=2,
                                                    transformer_hidden_size=64,
                                                    transformer_dropout=0.1)  # found in code, default value

BERT4REC_MODEL_CONFIG_MOVIELENS_20M = BERT4REC_MODEL_CONFIG_MOVIELENS_1M

BERT4REC_MODEL_CONFIG_BEAUTY = BERT4RecConfig(item_voc_size=-1,  # set it to the correct vocab size
                                              max_seq_length=50,
                                              num_transformer_heads=2,
                                              num_transformer_layers=2,
                                              transformer_hidden_size=64,
                                              transformer_dropout=0.1)

BERT4REC_MODEL_CONFIG_STEAM = BERT4REC_MODEL_CONFIG_BEAUTY



def build_dataset_config(dataset_path: str,
                         batch_size: int,
                         max_seq_length: int,
                         dataset_id: str,
                         item_column_name: str,
                         nip: bool = False
                         ) -> dict:
    dataset_config = {
        'csv_file': f"{dataset_path}/{dataset_id}.csv",
        'csv_file_index': f"{dataset_path}/{dataset_id}.idx",
        'delimiter': "\t",
        'item_column_name': item_column_name
    }

    if nip:
        dataset_config['nip_index_file'] = f"{dataset_path}/{dataset_id}.nip.idx"
    return {
        'dataset': dataset_config,
        'loader': {
            'batch_size': batch_size,
            'max_seq_length': max_seq_length
        }
    }


def build_model_config_bert4rec(item_vocab_size: int,
                                max_seq_length: int,
                                dropout: float = 0.1,
                                num_transformer_heads: int = 2,
                                num_transformer_layers: int = 2,
                                transformer_hidden_size: int = 16) -> Dict[str, Any]:
    config = BERT4RecConfig(item_voc_size=item_vocab_size,
                            max_seq_length=max_seq_length,
                            transformer_dropout=dropout,
                            num_transformer_heads=num_transformer_heads,
                            num_transformer_layers=num_transformer_layers,
                            transformer_hidden_size=transformer_hidden_size)
    return build_model_config_from_bert4rec_config(config)


def build_model_config_from_bert4rec_config(config: BERT4RecConfig
                                            ) -> Dict[str, Any]:
    return config.to_dict()


def _build_bert4rec_config_clinic():
    dataset_path = "/home/zoller_d1/dzptm/seq_reco/dataset"
    result_dir = "/home/zoller_d1/dzptm/seq_reco/runs/bert4rec"
    tokenizer_config = {
        'vocabulary': {
            'delimiter': "\t",
            'file': f"{dataset_path}/items.vocab"
        },
        'special_tokens': {
            'pad_token': "<PAD>",
            'mask_token': "<MASK>",
            'unk_token': "<UNK>"
        }
    }

    batch_size = 512

    item_vocab_size = 248  # TODO: find out, can this be injected by the tokenizer?
    config = copy.copy(BERT4REC_MODEL_CONFIG_MOVIELENS_20M)
    config.item_voc_size = item_vocab_size
    max_seq_length = config.max_seq_length

    model_config = build_model_config_from_bert4rec_config(config)

    module_config = {
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'beta_1': 0.99,
        'beta_2': 0.998,
        'batch_first': True,
        'metrics': {
            'recall': [1, 3, 5],
            'mrr': [1, 3, 5]
        }
    }

    item_column_name = "test"

    train_config = build_dataset_config(dataset_path, batch_size, max_seq_length, 'train', item_column_name)
    val_config = build_dataset_config(dataset_path, batch_size, max_seq_length, 'valid', item_column_name, nip=True)
    test_config = build_dataset_config(dataset_path, batch_size, max_seq_length, 'test', item_column_name, nip=True)

    datasets_config = {
        'train': train_config,
        'validation': val_config,
        'test': test_config
    }

    trainer_config = {
        'default_root_dir': result_dir,
        'checkpoints': {
            'save_top_k': 3,
            'monitor': "recall_at_5"
        }
    }

    bert4rec_config = {
        'tokenizer': tokenizer_config,
        'model': model_config,
        'module': module_config,
        'datasets': datasets_config,
        'trainer': trainer_config
    }

    with open('clinic_bert4rec_config.yaml', 'w') as outfile:
        yaml.dump(bert4rec_config, outfile)


if __name__ == '__main__':
    _build_bert4rec_config_clinic()
