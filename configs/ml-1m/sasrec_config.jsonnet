local base_path = "/ssd/ml-20m/";
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
            path: "/scratch/jane-doe-framework/experiments/ml-1m/sasrec"
        },
        pos_neg_data_sources: {
            parser: {
                item_column_name: "title"
            },
            loader: {
                batch_size: 64,
                max_seq_length: max_seq_length,
                num_workers: 10
            },
            path: base_path,
            file_prefix: file_prefix,
            split_type: "leave_one_out", // leave one out split for evaluation
        }
    },
    module: {
        type: "sasrec",
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
            transformer_hidden_size: 128,
            num_transformer_heads: 2,
            num_transformer_layers: 2,
            max_seq_length: max_seq_length,
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