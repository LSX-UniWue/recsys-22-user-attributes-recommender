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
            path: "/tmp/experiments/narm"
        },
        next_sequence_step_data_sources: {
            parser: {
                item_column_name: "item_id"
            },
            loader: {
                batch_size: 9,
                max_seq_length: max_seq_length
            },
            path: base_path + "ratio_split/",
            train_file_prefix: "example",
            validation_file_prefix: "example",
            test_file_prefix: "example"
        },
    },
    module: {
        type: "narm",
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
            item_embedding_size: 4,
            global_encoder_size: 128,
            global_encoder_num_layers: 1,
            embedding_dropout: 0.25,
            context_dropout: 0.25,
            batch_first: true
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
        logger: {
            type: "tensorboard"
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max'
        }
    }
}