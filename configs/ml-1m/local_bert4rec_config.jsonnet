local base_path = "../datasets/dataset/ml-1m/";
local loo_path = base_path + "loo/";
local output_path = "../dataset/ml-1m/exp/";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};

local file_prefix = 'ml-1m';

{
    datamodule: {
        dataset: "ml-1m",
        preprocessing: {
            extraction_directory: "/tmp/ml-1m/",
            output_directory: base_path,
            min_item_feedback: 0,
            min_sequence_length: 2
        }
    },
    templates: {
        unified_output: {
            path: output_path
        },
        mask_data_sources: {
            parser: {
                item_column_name: "title"
            },
            loader: {
                batch_size: 4,
                max_seq_length: max_seq_length
            },
            path: base_path,
            file_prefix: file_prefix,
            split_type: 'leave_one_out',
            mask_probability: 0.2,
            mask_seed: 42
        }
    },
    module: {
        type: "bert4rec",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
              sample_probability_file: loo_path + file_prefix + ".popularity.title.txt",
                num_negative_samples: 100,
                metrics: metrics
            }
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 1,
            num_transformer_layers: 1,
            transformer_hidden_size: 32,
            transformer_dropout: 0.2,
            project_layer_type: 'linear'
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
                    file: loo_path + file_prefix+".vocabulary.title.txt"
                }
            }
        },
    },
    trainer: {
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@10_ssampled(100)",
            save_top_k: 3,
            mode: 'max'
        },
        gpus: 0,
        max_epochs: 10,
        check_val_every_n_epoch: 50
    }
}
