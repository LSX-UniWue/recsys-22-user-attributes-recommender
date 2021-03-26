local base_path = "/ssd/ml-1m/";
local output_path = "/scratch/jane-doe-framework/experiments/ml-1m/bert4rec/";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};

local file_prefix = 'ml-1m';

{
    templates: {
        unified_output: {
            path: output_path
        },
        mask_data_sources: {
            parser: {
                item_column_name: "title"
            },
            loader: {
                batch_size: 64,
                max_seq_length: max_seq_length,
                num_workers: 8
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
                sample_probability_file: base_path + "ml-1m.popularity.title.txt",
                num_negative_samples: 100,
                metrics: metrics
            }
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 2,
            num_transformer_layers: 2,
            transformer_hidden_size: 128,
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
                    file: base_path + "ml-1m.vocabulary.title.txt"
                }
            }
        }
    },
    trainer: {
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@10_sampled(100)",
            save_top_k: 3,
            mode: 'max'
        },
        gpus: 1,
        max_epochs: 800,
        check_val_every_n_epoch: 10
    }
}
