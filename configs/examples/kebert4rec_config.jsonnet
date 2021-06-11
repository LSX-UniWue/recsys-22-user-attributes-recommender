local base_path = "../tests/example_dataset/";
local max_seq_length = 7;
local prefix = 'example';
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    rank: []
};
{
    templates: {
        unified_output: {
            path: "/tmp/experiments/bert4rec"
        },
        mask_data_sources: {
            loader: {
                batch_size: 9,
                num_workers: 0
            },
            path: base_path,
            file_prefix: prefix,
            mask_probability: 0.1,
            split_type: 'leave_one_out'
        }
    },
    module: {
        type: "kebert4rec",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + "example.popularity.item_id.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
            random_negative_sampled: {
                num_negative_samples: 2,
                metrics: metrics
            },
            fixed: {
                item_file: base_path + "example.relevant_items.item_id.txt",
                metrics: metrics
            }
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 1,
            num_transformer_layers: 1,
            transformer_hidden_size: 2,
            transformer_dropout: 0.1,
            additional_attributes: {
                attr_one: {
                    embedding_type: 'content_embedding',
                    vocab_size: 7
                }
            }
        }
    },
    features: {
        item: {
            column_name: "item_id",
            sequence_length: max_seq_length,
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
        },
        attr_one: {
            sequence_length: max_seq_length,
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
                vocabulary: {
                    file: base_path + "example.vocabulary.attr_one.txt"
                }
            }
        }
    },
    trainer: {
        loggers: {
            tensorboard: {},
            csv: {}
        },
        checkpoint: {
            monitor: "recall@5_fixed",
            save_top_k: 3,
            mode: 'max'
        },
        early_stopping: {
          monitor: 'recall@5_fixed',
          min_delta: 0.00,
          patience: 10,
          mode: 'max'
        },
        max_epochs: 5
    }
}
