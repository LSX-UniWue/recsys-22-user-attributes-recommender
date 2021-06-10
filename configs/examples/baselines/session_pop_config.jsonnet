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
            path: "/tmp/experiments/session_pop"
        },
        plain_training_next_item_test_and_validation_data_sources: {
            loader: {
                batch_size: 8
            },
            path: base_path + 'ratio_split/',
            file_prefix: prefix,
            split_type: 'ratio_split'
        }
    },
    module: {
        type: "session_pop",
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
    features: {
        item: {
            column_name: "item_id",
            sequence_length: max_seq_length
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
                    file: base_path + "example.vocabulary.item_id.txt"
                }
            }
        }
    },
    trainer: {
        max_epochs: 1,
        logger: {
            type: "tensorboard",
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        },
    }
}
