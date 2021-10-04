#local raw_dataset_path = "../tests/example_dataset/";
local dataset_path = "/mnt/c/Users/seife/work/minor/recommender/tests/example_dataset/";
local dataset = 'example';
local number_of_targets = 2;
local max_seq_length = 4;
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5]
};
{   datamodule: {
        dataset: dataset,
        template: {
            name: "sliding_window",
            split: "ratio_split",
            file_prefix: dataset,
            num_workers: 0,
            batch_size: 9,
            dynamic_padding: false,
            window_size: max_seq_length,
            number_target_interactions: number_of_targets
        },
        preprocessing: {
            output_directory: dataset_path,
            input_file_path: "/mnt/c/Users/seife/work/minor/recommender/tests/example_dataset/example.csv"
        }
    },
    templates: {
        unified_output: {
            path: "/tmp/experiments/caser"
        }
    },
    module: {
        type: "caser",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: dataset_path + "example.popularity.item_id.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
            fixed: {
                item_file: dataset_path + "example.relevant_items.item_id.txt",
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
                    file: dataset_path + "example.vocabulary.item_id.txt"
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
