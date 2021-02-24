local dataset = import 'dataset_base.libsonnet';

local base_path = "../tests/example_dataset/";

local cloze_processor = {
    type: "cloze",
    mask_probability: 0.2,
    only_last_item_mask_prob: 0.1,
    seed: 123456
};
local pos_neg_sampler = {
    type: "pos_neg",
    seed: 42
};


{
    data_sources: dataset.example_dataset(base_path, "session", train_processors=[pos_neg_sampler], test_processors=[], validation_processors=[]),
    module: {
        type: "caser",
        metrics: {
            full: {
                metrics: {
                    mrr: [1, 3, 5],
                    recall: [1, 3, 5],
                    ndcg: [1, 3, 5]
                }
            },
            sampled: {
                sample_probability_file: base_path + "popularity.txt",
                num_negative_samples: 2,
                metrics: {
                    mrr: [1, 3, 5],
                    recall: [1, 3, 5],
                    ndcg: [1, 3, 5]
                }
            },
            fixed: {
                item_file: base_path + "relevant_items.txt",
                metrics: {
                    mrr: [1, 3, 5],
                    recall: [1, 3, 5],
                    ndcg: [1, 3, 5]
                }
            }
        },
        model: {
            dropout: 0.1,
            max_seq_length: 5,
            embedding_size: 4,
            num_vertical_filters: 2,
            num_horizontal_filters: 1,
            fc_activation_fn: "relu",
            conv_activation_fn: "relu",
        }
    },
    tokenizers: dataset.tokenizer(base_path),
    trainer: {
        logger: {
            type: "tensorboard",
            save_dir: "/tmp/bert4rec",
            name: "bert4rec",
            version: ""
        },
        checkpoint: {
            monitor: "recall@5",
            save_top_k: 3,
            mode: 'max',
            dirpath: "/tmp/bert4rec/checkpoints"
        }
    }
}