local base_path = "../tests/example_dataset/";
local max_seq_length = 7;
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5]
};
{
    next_sequence_step_data_sources: {
        parser: {
            item_column_name: "item_id",
            item_separator: ' + '
        },
        batch_size: 9,
        max_seq_length: max_seq_length,
        path: base_path,
        validation_file_prefix: "train",
        test_file_prefix: "train",
        seed: 123456
    },
    module: {
        type: "rnn",
        metrics: {
            full: {
                metrics: metrics
            },
            sampled: {
                sample_probability_file: base_path + "popularity.txt",
                num_negative_samples: 2,
                metrics: metrics
            },
            fixed: {
                item_file: base_path + "relevant_items.txt",
                metrics: metrics
            }
        },
        model: {
            cell_type: "lstm",
            item_embedding_dim: 4,
            hidden_size: 4,
            num_layers: 1,
            dropout: 0.0
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
                        file: base_path + "vocab.txt"
                    }
                }
            }
        },
    trainer: {
        logger: {
            type: "tensorboard",
            save_dir: "/tmp/caser",
            name: "caser",
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max',
            dirpath: "/tmp/caser/checkpoints"
        }
    }
}