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
            path: "/tmp/experiments/bpr"
        },
        pos_neg_data_sources: {
            parser: {
                item_column_name: "item_id",
                additional_attributes: {
                    user_id: {
                        type: "int",
                        sequence: false,
                    }
                }

            },
            loader: {
                batch_size: 9,
                max_seq_length: max_seq_length
            },
            path: base_path + "ratio_split/",
            file_prefix: "example",
            seed: 123456
        }
    },
    module: {
        type: "bpr",
        num_users: 1000,
        embedding_size: 16,
        regularization_factor: 0.1,
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
                    file: base_path + prefix +".vocabulary.item_id.txt"
                }
            }
        },
        user: {
            tokenizer: {
                special_tokens: {

                },
                vocabulary: {
                    file: base_path + prefix + ".vocabulary.user_id.txt"
                }
            }
        }
    },
    trainer: {
        max_epochs: 1,
        num_sanity_val_steps: 0,
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
