local base_path = "datasets/dataset/ml-1m/";
local output_path = "dataset/datassts/ml-1m/exp/";
local max_seq_length = 10;
local metrics =  {
    mrr: [1, 5, 10],
    recall: [1, 5, 10],
    ndcg: [1, 5, 10]
};

local file_prefix = 'ml-1m';

local additional_attributes_list = ['genres'];

{
    templates: {
        unified_output: {
            path: output_path
        },
        mask_data_sources: {
            loader: {
                batch_size: 4,
                num_workers: 0
            },
            path: base_path,
            train_file_prefix: file_prefix,
            validation_file_prefix: file_prefix,
            test_file_prefix: file_prefix,
            split_type: 'leave_one_out',
            mask_probability: 0.2,
            mask_seed: 42,
            mask_additional_attributes: additional_attributes_list
        }
    },
    module: {
        type: "kebert4rec",
        additional_attributes: additional_attributes_list,
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + "loo/" + file_prefix + ".popularity.title.txt",
                num_negative_samples: 100,
                metrics: metrics
            }
        },
        model: {
            max_seq_length: max_seq_length,
            max_seq_step_length: 5,
            num_transformer_heads: 1,
            num_transformer_layers: 1,
            transformer_hidden_size: 32,
            transformer_dropout: 0.2,
            project_layer_type: 'linear',
            additional_attributes: {
                genres: {
                    embedding_type: 'linear_upscale',
                    vocab_size: 301
                }
            }
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
                    file: base_path + "loo/" + file_prefix + ".vocabulary.title.txt"
                }
            }
        },
        genres: {
            type: "strlist",
            delimiter: "|",
            sequence_length: max_seq_length,
            max_sequence_step_length: 5,
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
                vocabulary: {
                    file: base_path + "loo/" + file_prefix + ".vocabulary.genres.txt"
                }
            }
        },
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
        max_epochs: 10,
        check_val_every_n_epoch: 50
    }
}
