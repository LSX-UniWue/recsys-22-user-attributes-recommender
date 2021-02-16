{
    parser: {
        item_column_name: "item_id"
    },
    loader: {
        batch_size: 2,
        max_seq_length: 5
    },
    datasets: {
        test: {
            dataset: {
                csv_file: "../tests/example_dataset/train.csv",
                csv_file_index: "../tests/example_dataset/train.idx",
                parser: $['parser'],
                nip_index_file: "../tests/example_dataset/train.nip.idx"
            },
            loader: $['loader']
        },
        validation: {
            dataset: {
                csv_file: "../tests/example_dataset/train.csv",
                csv_file_index: "../tests/example_dataset/train.idx",
                parser: $['parser'],
                nip_index_file: "../tests/example_dataset/train.nip.idx"
            },
            loader: $['loader']
        },
        train: {
            dataset: {
                csv_file: "../tests/example_dataset/train.csv",
                csv_file_index: "../tests/example_dataset/train.idx",
                parser: $['parser'],
            },
            loader: $['loader']
        }
    },
    model: {
        item_vocab_size: 13,
        max_seq_length: 5,
        num_transformer_heads: 1,
        num_transformer_layers: 1,
        transformer_hidden_size: 2,
        transformer_dropout: 0.1
    },
    module: {
        mask_probability: 0.9,
        metrics: {
            mrr: [-1, -3, -5],
            recall:[-1, -3, -5]
        }
    },
    tokenizer: {
        special_tokens: {
            pad_token: "<PAD>",
            mask_token: "<MASK>",
            unk_token: "<UNK>"
        },
        vocabulary: {
            delimiter: "\t",
            file: "../tests/example_dataset/vocab.txt"
        }
    },
    trainer: {
        checkpoint: {
            monitor: "recall_at_5",
            save_top_k: 3
        }
    }
}
