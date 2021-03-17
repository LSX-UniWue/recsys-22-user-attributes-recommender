local base_path = "/home/dallmann/uni/research/repositories/dataset/conventions/ml-20m_3_0_0/";
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
            path: "/tmp/bert4rec_new"
        },
        mask_data_sources: {
            parser: {
                item_column_name: "title"
            },
            loader: {
                batch_size: 128,
                max_seq_length: max_seq_length
            },
            path: base_path,
            file_prefix: file_prefix,
            split_type: "leave_one_out", // leave one out split
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
                sample_probability_file: base_path + "ml-20m.popularity.title.txt",
                num_negative_samples: 1000,
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
                    file: base_path + "ml-20m.vocabulary.title.txt"
                }
            }
        }
    },
    trainer: {
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@10/sampled(100)",
            save_top_k: 3,
            mode: 'max'
        },
        gpus: 1,
        max_epochs: 800,
        #accelerator: "ddp",
        limit_test_batches: 10,
        check_val_every_n_epoch: 100

    }
}
