import yaml


def build_config(dataset_path: str,
                 batch_size: int,
                 max_seq_length:int,
                 dataset_id:str,
                 nip: bool = False
                 ) -> dict:
    dataset_config = {
        'csv_file': "{}/{}.csv".format(dataset_path, dataset_id),
        'csv_file_index': "{}/{}.idx".format(dataset_path, dataset_id),
        'delimiter': "\t",
        'item_column_name': "item_id"
    }

    if nip:
        dataset_config['nip_index_file'] = "{}/{}.nip.idx".format(dataset_path, dataset_id)
    return {
        'dataset': dataset_config,
        'loader': {
            'batch_size': batch_size,
            'max_seq_length': max_seq_length
        }
    }


if __name__ == '__main__':
    #dataset_path = "/home/dallmann/uni/research/dota/datasets/small/splits"
    dataset_path = "/Users/nosebrain/Desktop/small"
    tokenizer_config = {
        'vocabulary': {
            'delimiter': "\t",
            'file': "{}/items.vocab".format(dataset_path)
        },
        'special_tokens': {
            'pad_token': "<PAD>"
        }
    }

    max_seq_length = 2047
    batch_size = 4

    model_config = {
        'transformer_hidden_size': 16,
        'num_transformer_heads': 1,
        'num_transformer_layers': 1,
        'item_vocab_size': 248,
        'max_seq_length': max_seq_length,
        'dropout': 0.1
    }

    module_config = {
        'batch_size': batch_size,
        'learning_rate': 0.001,
        'beta_1': 0.99,
        'beta_2': 0.998,
        'batch_first': True,
        'metrics_k': [1, 3, 5, 10]
    }

    train_config = build_config(dataset_path, batch_size, max_seq_length, 'train')
    val_config = build_config(dataset_path, batch_size, max_seq_length, 'valid', nip=True)
    test_config = build_config(dataset_path, batch_size, max_seq_length, 'test', nip=True)

    datasets_config = {
        'train': train_config,
        'validation' : val_config,
        'test': test_config
    }

    trainer_config = {
        'limit_train_batches': 10,
        'limit_val_batches': 10,
        'default_root_dir': '/tmp/checkpoints',
        'checkpoints': {
            'save_top_k': 3,
            'monitor': "recall_at_5"
        }
    }

    sas_config = {
        'tokenizer': tokenizer_config,
        'model': model_config,
        'module': module_config,
        'datasets': datasets_config,
        'trainer': trainer_config
    }

    with open('sas_config.yaml', 'w') as outfile:
        yaml.dump(sas_config, outfile)