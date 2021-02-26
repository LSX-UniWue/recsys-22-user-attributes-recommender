local base_path = "../dataset/dataset/ml-1m_3_5_5/";
local max_seq_length = 50;
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5]
};

local file_prefix = 'ml-1m';

{
    templates: {
        unified_output: {
            path: "../dataset/dataset/experiments/ml-1m/gru"
        },
        next_sequence_step_data_sources: {
            parser: {
                item_column_name: "title"
            },
            loader: {
                batch_size: 16,
                max_seq_length: max_seq_length
            },
            path: base_path,
            train_file_prefix: file_prefix,
            validation_file_prefix: file_prefix,
            test_file_prefix: file_prefix,
            leave_one_out: true,
        }
    },
    module: {
        type: "rnn",
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
            cell_type: "gru",
            item_embedding_dim: 64,
            hidden_size: 64,
            num_layers: 2,
            dropout: 0.2
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
                    delimiter: "\t",
                    file: base_path + "vocab_title.txt"
                }
            }
        }
    },
    trainer: {
        logger: {
            type: "tensorboard"
        },
        checkpoint: {
            monitor: "recall@5/sampled(100)",
            save_top_k: 3,
            mode: 'max'
        },
        max_epochs: 100
    }
}