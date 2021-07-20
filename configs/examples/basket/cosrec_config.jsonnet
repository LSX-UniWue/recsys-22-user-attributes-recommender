local base_path = "../tests/example_dataset/";
local max_seq_length = 5;
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5]
};
{
    templates: {
        unified_output: {
            path: "/tmp/experiments/cosrec"
        },
        pos_neg_data_sources: {
            loader: {
                batch_size: 9
            },
            path: base_path + "ratio_split/",
            file_prefix: "example",
            seed: 123456
        }
    },
    module: {
        type: "cosrec",
        learning_rate: 0.001,
        weight_decay: 0.01,
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
        model: {
            user_vocab_size: 0,
            max_seq_length: max_seq_length,
            # chosen from {10, 20, 30, 50, 100}, ML-1M: 50, Gowalla: 100
            embed_dim: 50,
            block_num: 2,
            block_dim:[128,256],
            fc_dim: 150,
            activation_function: 'relu',
            dropout: 0.5,

        }
    },
    features: {
        item: {
            column_name: "item_id",
            type: "strlist",
            delimiter: " + ",
            sequence_length: max_seq_length,
            max_sequence_step_length: 5,
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
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        }
    }
}