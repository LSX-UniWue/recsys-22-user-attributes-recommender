local base_path = "/scratch/jane-doe-framework/students/ml-1m/ml-1m/";
local output_path = "/scratch/jane-doe-framework/experiments/ml-1m/cosrec/";
local max_seq_length = 200;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};
local dataset = 'ml-1m';
{

    datamodule: {
        dataset: dataset,
        template: {
            name: "par_pos_neg",
            split: "leave_one_out",
            path: base_path,
            file_prefix: dataset,
            num_workers: 4,
            batch_size: 64
            seed: 123456,
            t: 1
        },
        preprocessing: {
            extraction_directory: "/tmp/ml-1m/",
            output_directory: base_path,
            min_item_feedback: 4,
            min_sequence_length: 4,
        }
    },
    templates: {
        unified_output: {
            path: output_path
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
                    # Inferred by the datamodule
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