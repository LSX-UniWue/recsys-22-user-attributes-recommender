{
    parser: {
        item_column_name: "item_id"
    },
    loader: {
        batch_size: 2,
        max_seq_length: 5
    },
    data_sources: {
        test: {
            loader: {
                dataset: {
                    type: "nextit",
                    csv_file: "../tests/example_dataset/train.csv",
                    csv_file_index: "../tests/example_dataset/train.idx",
                    parser: $['parser'],
                    nip_index_file: "../tests/example_dataset/train.nip.idx",
                    processors: [
                        {
                            type: "tokenizer"
                        },
                        {
                            type: "last_item"
                        }
                    ]
                }
            } + $['loader']
        },
        validation: {
            loader: {
                dataset: {
                    type: "nextit",
                    csv_file: "../tests/example_dataset/train.csv",
                    csv_file_index: "../tests/example_dataset/train.idx",
                    parser: $['parser'],
                    nip_index_file: "../tests/example_dataset/train.nip.idx",
                    processors: [
                        {
                            type: "tokenizer"
                        },
                        {
                            type: "last_item"
                        }
                    ]
                }
            } + $['loader']
        },
        train: {
            loader: {
                dataset: {
                    type: "session",
                    csv_file: "../tests/example_dataset/train.csv",
                    csv_file_index: "../tests/example_dataset/train.idx",
                    parser: $['parser'],
                    processors:  [
                        {
                            type: "tokenizer"
                        },
                        {
                            type: "cloze",
                            mask_probability: 0.2,
                            only_last_item_mask_prob: 0.1,
                            seed: 123456
                        }
                    ]
                },
            } + $['loader']
        }
    },
    module: {
        type: "bert4rec",
        metrics: {
            full: {
                metrics: {
                    #mrr: [1, 3, 5],
                    recall: [1, 3, 5],
                    #ndcg: [1, 3, 5]
                }
            },
            sampled: {
                sample_probability_file: "../tests/example_dataset/popularity.txt",
                num_negative_samples: 2,
                metrics: {
                    mrr: [1, 3, 5],
                    recall: [1, 3, 5],
                    ndcg: [1, 3, 5]
                }
            },
            fixed: {
                item_file: "../tests/example_dataset/relevant_items.txt",
                metrics: {
                    mrr: [1, 3, 5],
                    recall: [1, 3, 5],
                    ndcg: [1, 3, 5]
                }
            }
        },
        model: {
            max_seq_length: 5,
            num_transformer_heads: 1,
            num_transformer_layers: 1,
            transformer_hidden_size: 2,
            transformer_dropout: 0.1
        }
    },
    tokenizers: {
        item: {
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
            }
        }
    },

    trainer: {
        logger: {
            type: "tensorboard",
            save_dir: "/tmp/bert4rec",
            name: "bert4rec",
            version: ""
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max',
            dirpath: "/tmp/bert4rec/checkpoints"
        }
    }
}
