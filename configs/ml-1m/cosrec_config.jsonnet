local base_path = "/scratch/jane-doe-framework/students/ml-1m/ml-1m/";
local output_path = "/scratch/jane-doe-framework/experiments/ml-1m/cosrec/";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};
local loo_path = base_path + "loo/";
local file_prefix = 'ml-1m';
{
    templates: {
        unified_output: {
            path: output_path
        },
        par_pos_neg_data_sources: {
            loader: {
                batch_size: 4,
                num_workers: 4
            },
            path: base_path,
            file_prefix: "ml-1m",
            split_type: "leave_one_out",
            seed: 123456,
            t: 1
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

        },
        model: {
            user_vocab_size: 0,
            max_seq_length: max_seq_length,
            embed_dim: 50,
            block_num: 2,
            block_dim: [128, 256],
            fc_dim: 150,
            activation_function: 'relu',
            dropout: 0.5
        }
    },
    features: {
        item: {
            column_name: "title",
            sequence_length: max_seq_length,
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
                vocabulary: {
                    file:  loo_path + file_prefix + ".vocabulary.title.txt"
                }
            }
        }
    },
    trainer: {
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@10",
            save_top_k: 3,
            mode: 'max'
        },
        max_epochs: 800,
        check_val_every_n_epoch: 100
    }
}