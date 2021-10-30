local base_path = "/mnt/c/Users/seife/work/recommender/tests/example_dataset/";
local output_path = '/tmp/experiments/pop';
local max_seq_length = 7;
local dataset = 'example';
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5]
};
{
     datamodule: {
        dataset: dataset,
        data_sources: {
            split: "leave_one_out",
            file_prefix: dataset,
            num_workers: 0,
            train: {
                type: "session",
                processors: []
            },
            validation: {
                type: "next_item",
                processors: [
                    {
                        "type": "target_extractor"
                    },
                ]
            },
            test: {
             type: "next_item",
                processors: [
                    {
                        "type": "target_extractor"
                    },
                ]
            }
        },
        preprocessing: {
            input_file_path: base_path + "example.csv",
            output_directory: base_path
        }
    },
    templates: {
        unified_output: {
            path: output_path
        },
     },
    module: {
        type: "pop",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + dataset + ".popularity.item_id.txt",
                num_negative_samples: 2,
                metrics: metrics
            }
        },
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
                }
            }
        }
    },
    trainer: {
        max_epochs: 1,
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        },
    }
}
