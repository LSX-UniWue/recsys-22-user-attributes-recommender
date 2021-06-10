local base_path = "../tests/example_dataset/";
local max_seq_length = 7;
local metrics =  {
    mrr: [1, 3, 5],
    recall: [1, 3, 5],
    ndcg: [1, 3, 5]
};
{
    templates: {
        unified_output: {
            path: "/tmp/experiments/sasrec"
        },
        pos_neg_data_sources: {
            loader: {
                batch_size: 9
            },
            path: base_path + "ratio-0.8_0.1_0.1/",
            file_prefix: "example",
            seed: 123456
        }
    },
    module: {
        type: "sasrec",
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
            transformer_hidden_size: 4,
            num_transformer_heads: 2,
            num_transformer_layers: 1,
            max_seq_length: max_seq_length,
            transformer_dropout: 0.1
        }
    },
    features: {
        item: {
            column_name: "item_id",
            sequence_length: max_seq_length
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