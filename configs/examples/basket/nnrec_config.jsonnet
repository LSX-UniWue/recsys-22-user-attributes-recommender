local base_path = "../tests/example_dataset/";
local max_seq_length = 5;
local prefix = 'example';
local metrics =  {
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    f1: [1, 3, 5]
};
{
    templates: {
        unified_output: {
            path: "/tmp/experiments/nnrec_basket"
        },
        next_sequence_step_data_sources: {
            loader: {
                batch_size: 9,
                dynamic_padding: false
            },
            path: base_path,
            file_prefix: prefix,
            split_type: 'leave_one_out'
        }
    },
    module: {
        type: "nnrec",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + "example.popularity.item_id.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
            fixed: {
                item_file: base_path + "example.relevant_items.item_id.txt",
                metrics: metrics
            }
        },
        model: {
            user_vocab_size: 0,
            item_embedding_size: 4,
            hidden_size: 16,
            user_embedding_size: 0,
            embedding_pooling_type: "mean",
            max_sequence_length: max_seq_length
        }
    },
    features: {
        item: {
            column_name: "item_id",
            type: "strlist",
            delimiter: " + ",
            sequence_length: max_seq_length,
            max_sequence_step_length: 5,
            tokenizer: {
                special_tokens: {
                    pad_token: "<PAD>",
                    mask_token: "<MASK>",
                    unk_token: "<UNK>"
                },
                vocabulary: {
                    file: base_path + "example.vocabulary.item_id.txt"
                }
            }
        }
    },
    trainer: {
        loggers: {
            tensorboard: {}
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        }
    }
}