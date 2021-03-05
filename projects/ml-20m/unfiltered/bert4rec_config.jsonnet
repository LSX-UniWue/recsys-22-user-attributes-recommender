local base_path = "/ssd/ml-20m/";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};

local file_prefix = 'ml-20m';

{
    templates: {
        unified_output: {
            path: "/scratch/jane-doe-framework/experiments/ml-20m/bert4rec"
        },
        mask_data_sources: {
            parser: {
                item_column_name: "title"
            },
            loader: {
                batch_size: 64,
                max_seq_length: max_seq_length
            },
            path: base_path,
            train_file_prefix: file_prefix,
            validation_file_prefix: file_prefix,
            test_file_prefix: file_prefix,
            next_step_type: "loo", // leave one out
            mask_probability: 0.2,
            seed: 42
        }
    },
    module: {
        type: "bert4rec",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + "popularity.txt",
                num_negative_samples: 100,
                metrics: metrics
            }
        },
        model: {
            max_seq_length: max_seq_length,
            num_transformer_heads: 2,
            num_transformer_layers: 2,
            transformer_hidden_size: 256,
            transformer_dropout: 0.2
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
        logger: {
            type: "tensorboard",
        },
        checkpoint: {
            monitor: "recall@10/sampled(100)",
            save_top_k: 3,
            mode: 'max'
        },
        gpus: 8,
        max_epochs: 800,
        accelerator: "ddp",
        check_val_every_n_epoch: 50
    }
}
