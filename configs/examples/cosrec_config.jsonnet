local base_path = "../tests/example_dataset/";
local max_seq_length = 10;
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5]
};
local dataset = 'example';
{   datamodule: {
        cache_path: "/tmp/ssd",
        dataset: dataset,
        template: {
            name: "par_pos_neg",
            split: "ratio_split",
            file_prefix: dataset,
            num_workers: 0,
            batch_size: 9,
            number_negative_items: 1,
            number_positive_items: 1
        },
        preprocessing: {
            input_file_path: base_path+"example.csv",
            output_directory: base_path
        }
    },
    templates: {
        unified_output: {
            path: "/tmp/experiments/cosrec"
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
            embed_dim: 50,
            block_num: 1,
            block_dim: [128],
            fc_dim: 150,
            activation_function: 'relu',
            dropout: 0.5
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
