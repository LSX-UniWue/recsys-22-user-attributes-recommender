{
    parser: {
        item_column_name: "item_id",
        item_separator: " + "
    },
    loader: {
        batch_size: 2,
        max_seq_length: 5,
        max_seq_step_length: 5
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
                        }
                    ]
                },
            } + $['loader']
        }
    },
    module: {
        type: "dream",
        metrics: {
            full: {
                metrics: {
                    mrr: [1, 3, 5],
                    recall: [1, 3, 5],
                    ndcg: [1, 3, 5]
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
            cell_type: "gru",
            item_embedding_dim: 4,
            hidden_size: 4,
            num_layers: 5,
            dropout: 0.2,
            embedding_pooling_type: "mean",
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
            save_dir: "/tmp/dream",
            name: "bert4rec",
            version: ""
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max',
            dirpath: "/tmp/dream/checkpoints"
        }
    }
}