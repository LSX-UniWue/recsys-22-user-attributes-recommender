local base_path = "../tests/example_dataset/";
local max_seq_length = 7;
local prefix = 'example';
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5]
};
{
    templates: {
        unified_output: {
            path: "/tmp/experiments/pop"
        },
    },
    data_sources: {
        train: {
            loader: {
                batch_size: 8,
                max_seq_length: max_seq_length,
                dataset: {
                    type: 'session',
                    csv_file: base_path + 'ratio_split/example.train.csv',
                    csv_file_index: base_path + 'ratio_split/example.train.session.idx',
                    parser: {
                        item_column_name: 'item_id',
                        delimiter: '\t'
                    },
                    processors: [
                        {
                            type: 'tokenizer'
                        }],
                    },
            }
        },
        validation: {
            loader: {
                batch_size: 8,
                max_seq_length: max_seq_length,
                dataset: {
                    type: 'sequence_position',
                    csv_file: base_path + 'ratio_split/example.validation.csv',
                    csv_file_index: base_path + 'ratio_split/example.validation.nextitem.idx',
                    nip_index_file: self.csv_file_index,
                    parser: {
                        item_column_name: 'item_id',
                        delimiter: '\t'
                    },
                    processors: [
                        {
                            type: 'tokenizer'
                        }],
                    },
            }
        },
        test: {
            loader: {
                batch_size: 8,
                max_seq_length: max_seq_length,
                dataset: {
                    type: 'sequence_position',
                    csv_file: base_path + 'ratio_split/example.test.csv',
                    csv_file_index: base_path + 'ratio_split/example.test.nextitem.idx',
                    nip_index_file: self.csv_file_index,
                    parser: {
                        item_column_name: 'item_id',
                        delimiter: '\t'
                    },
                    processors: [
                        {
                            type: 'tokenizer'
                        }],
                    },
            }
        }
    },
    module: {
        type: "pop",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + "example.popularity.item_id.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
            fixed: {
                item_file: base_path + "example.relevant_items.item_id.txt",
                metrics: metrics
            }
        },
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
                    file: base_path + "example.vocabulary.item_id.txt"
                }
            }
        }
    },
    trainer: {
        logger: {
            type: "tensorboard",
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        },
        early_stopping: {
          monitor: 'recall@5',
          min_delta: 0.00,
          patience: 10,
          mode: 'max'
        }
    }
}
