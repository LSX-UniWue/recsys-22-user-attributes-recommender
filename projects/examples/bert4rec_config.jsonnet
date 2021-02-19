{
    parser: {
        item_column_name: "item_id"
    },
    loader: {
        batch_size: 2,
        max_seq_length: 5
    },
    datasets: {
    # TODO configure processors everywhere
        test: {
            type: "nextit",
            dataset: {
                csv_file: "../tests/example_dataset/train.csv",
                csv_file_index: "../tests/example_dataset/train.idx",
                parser: $['parser'],
                nip_index_file: "../tests/example_dataset/train.nip.idx",
                processors: [
                    {
                        type: "cloze",
                        mask_probability: 0.2,
                        only_last_item_mask_prob: 0.1,
                        seed: 123456
                    },
                    {
                        type: "cloze",
                        mask_probability: 0.2,
                        only_last_item_mask_prob: 0.1,
                        seed: 123456
                    }
                ]
            },
            loader: $['loader']
        },
        validation: {
            type: "nextit",
            dataset: {
                csv_file: "../tests/example_dataset/train.csv",
                csv_file_index: "../tests/example_dataset/train.idx",
                parser: $['parser'],
                nip_index_file: "../tests/example_dataset/train.nip.idx"
            },
            loader: $['loader']
        },
        train: {
            type: "session",
            dataset: {
                csv_file: "../tests/example_dataset/train.csv",
                csv_file_index: "../tests/example_dataset/train.idx",
                parser: $['parser'],
            },
            loader: $['loader']
        }
    },
    module: {
        type: "bert4rec",
        mask_probability: 0.9,
        metrics: {
            full: {
                metrics: {
                    mrr: [1, 3, 5],
                    recall: [1, 3, 5]
                }
            },
            sampled: {
                sample_probability_file: "../tests/example_dataset/popularity.txt",
                num_negative_samples: 100,
                metrics: {
                    mrr: [1, 3, 5],
                    recall:[1, 3, 5]
                }
            }
        },
        model: {
            item_vocab_size: 13,
            max_seq_length: 5,
            num_transformer_heads: 1,
            num_transformer_layers: 1,
            transformer_hidden_size: 2,
            transformer_dropout: 0.1
        }
    },
    tokenizers: {
        item: {
            special_tokens: {
                pad_token: "<PAD>",
                mask_token: "<MASK>",
                unk_token: "<UNK>"
            },
            vocabulary: {
                delimiter: "\t",
                file: "../tests/example_dataset/vocab.txt"
            }
        }
    },

    trainer: {
        checkpoint: {
            monitor: "recall_at_5",
            save_top_k: 3
        }
    }
}
