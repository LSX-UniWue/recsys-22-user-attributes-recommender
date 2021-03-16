local base_path = "../tests/example_dataset/";
local max_seq_length = 7;
local metrics =  {
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    f1: [1, 3, 5]
};
{
    templates: {
        unified_output: {
            path: "/tmp/experiments/caser_basket"
        },
        pos_neg_data_sources: {
            parser: {
                item_column_name: "item_id",
                item_separator: ' + '
            },
            loader: {
                batch_size: 9,
                max_seq_length: max_seq_length,
                max_seq_step_length: 5
            },
            path: base_path,
            validation_file_prefix: "train",
            test_file_prefix: "train",
            seed: 123456
        }
    },
    module: {
        type: "caser",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + "popularity.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
            fixed: {
                item_file: base_path + "relevant_items.txt",
                metrics: metrics
            }
        },
        model: {
            dropout: 0.1,
            max_seq_length: max_seq_length,
            embedding_size: 4,
            num_vertical_filters: 2,
            num_horizontal_filters: 1,
            fc_activation_fn: "relu",
            conv_activation_fn: "relu",
            embedding_pooling_type: 'mean'
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
                        file: base_path + "vocab.txt"
                    }
                }
            }
        },
    trainer: {
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "f1@5",
            save_top_k: 3,
            mode: 'max'
        }
    }
}