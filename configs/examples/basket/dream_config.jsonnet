local base_path = "../tests/example_dataset/";
local max_seq_length = 7;
local metrics =  {
    recall: [1, 3, 5],
    ndcg: [1, 3, 5],
    f1: [1, 3, 5]
};
{
    templates: {
        unified_output: {
            path: "/tmp/experiments/dream_basket"
        },
        pos_neg_data_sources: {
            parser: {
                item_column_name: "item_id",
                item_separator: ' + '
            },
            loader: {
                batch_size: 9,
                max_seq_length: max_seq_length,
                max_seq_step_length: 5
            },
            file_prefix: "example",
            path: base_path + "ratio_split/",
            seed: 42
        }
    },
    module: {
        type: "dream",
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
            cell_type: "lstm",
            item_embedding_dim: 4,
            hidden_size: 4,
            num_layers: 1,
            dropout: 0.0,
            embedding_pooling_type: 'mean'
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