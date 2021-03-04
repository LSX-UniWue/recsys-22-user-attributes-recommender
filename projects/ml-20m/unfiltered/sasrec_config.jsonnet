local base_path = "/scratch/jane-doe-framework/datasets/ml-20m_3_0_0/";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};
{
    templates: {
        unified_output: {
            path: "/scratch/jane-doe-framework/experiments/ml-20m/sasrec"
        },
        pos_neg_data_sources: {
            parser: {
                item_column_name: "title"
            },
            loader: {
                batch_size: 64,
                max_seq_length: max_seq_length,
                num_workers: 4
            },
            path: base_path,
            validation_file_prefix: "train",
            test_file_prefix: "train",
            seed: 123456,
            split_type: "loo", // leave one out split for evaluation
        }
    },
    module: {
        type: "sasrec",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + "popularity.txt",
                num_negative_samples: 2,
                metrics: metrics
            }
        },
        model: {
            transformer_hidden_size: 64,
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
                    file: base_path + "vocab.txt"
                }
            }
        }
    },
    trainer: {
        logger: {
            type: "tensorboard"
        },
        checkpoint: {
            monitor: "recall@10/sampled(100)",
            save_top_k: 3,
            mode: 'max'
        },
        gpus: 8,
        max_epochs: 800,
        accelerator: "ddp",
        check_val_every_n_epoch: 100
    }
}